use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::mpsc;
use tokio::time::Instant;

/// How long the kernel gets to unwind after an interrupt before it is killed.
const KILL_GRACE: Duration = Duration::from_secs(2);
const STDERR_TAIL: usize = 8192;

/// Everything a cell can reach outside the kernel process.
#[async_trait]
pub trait Host: Send + Sync {
    async fn call(&self, name: &str, args: Value) -> Result<Value, String>;
}

#[derive(Debug)]
pub struct ExecOutcome {
    pub ok: bool,
    pub repr: String,
    pub traceback: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Frame {
    Ready,
    Stream {
        id: String,
        text: String,
    },
    HostRequest {
        req_id: String,
        /// Which cell asked, empty when a thread the cell spawned did.
        #[serde(default)]
        cell: String,
        #[serde(rename = "fn")]
        func: String,
        args: Value,
    },
    Done {
        id: String,
        status: String,
        #[serde(default)]
        repr: String,
        #[serde(default)]
        traceback: String,
    },
}

/// What the kernel is handed back after `env_clear`. A toolchain is found through `PATH`
/// and configured under `HOME`, so a cell that cannot see them cannot run `git`, `cargo`
/// or anything else the machine has — while a key held in any other variable stays on
/// this side of the pipe, which is the part that matters. `HOME` opens no door the cell
/// did not already have: the filesystem is reachable by absolute path regardless.
const MACHINE_ENV: [&str; 4] = ["PATH", "HOME", "LANG", "TMPDIR"];

fn machine_env() -> Vec<(String, String)> {
    let mut env: Vec<(String, String)> = MACHINE_ENV
        .iter()
        .filter_map(|name| Some((name.to_string(), std::env::var(name).ok()?)))
        .collect();
    if !env.iter().any(|(name, _)| name == "PATH") {
        env.push(("PATH".into(), "/usr/bin:/bin".into()));
    }
    env
}

/// Reaches a running cell from outside the `exec` that is awaiting it.
#[derive(Clone)]
pub struct Interrupter(mpsc::UnboundedSender<String>);

impl Interrupter {
    pub fn raise(&self) {
        let _ = self.0.send(encode(&interrupt()));
    }
}

/// A frame as it goes on the wire: one JSON object, one line. Everything written through
/// [`line_writer`] is framed here, whichever pipe it is headed down.
pub fn encode<T: Serialize>(frame: &T) -> String {
    format!(
        "{}\n",
        serde_json::to_string(frame).expect("a frame is plain data")
    )
}

fn interrupt() -> Value {
    json!({ "type": "interrupt" })
}

/// Hands a child's pipe to a task of its own, so a line can be written while something
/// else is being read back and two writers can never interleave inside one line.
pub fn line_writer(
    mut pipe: impl AsyncWrite + Unpin + Send + 'static,
) -> mpsc::UnboundedSender<String> {
    let (writes, mut outbox) = mpsc::unbounded_channel::<String>();
    tokio::spawn(async move {
        while let Some(line) = outbox.recv().await {
            if pipe.write_all(line.as_bytes()).await.is_err() || pipe.flush().await.is_err() {
                return;
            }
        }
    });
    writes
}

pub struct Kernel {
    child: Child,
    writes: mpsc::UnboundedSender<String>,
    frames: mpsc::Receiver<Frame>,
    stderr: Arc<Mutex<String>>,
}

impl Kernel {
    /// `fns` are the names bound in the cell namespace; each one round-trips to [`Host::call`].
    /// `skills` are the roots a cell imports a skill's package from, in the order they are
    /// preferred: the caller decides where they are, the kernel only puts them on the path.
    pub async fn start(
        python: impl AsRef<OsStr>,
        kernel_dir: &Path,
        skills: &[PathBuf],
        fns: &[&str],
    ) -> Result<Kernel> {
        // Joining fails only when a directory contains the separator itself, and there is
        // no value to fall back to then: python would split that same string in the same
        // place. Said outright, because the symptom is a shim that cannot be imported.
        let import_path = std::env::join_paths(
            std::iter::once(kernel_dir.to_path_buf()).chain(skills.iter().cloned()),
        )
        .with_context(|| format!("{} cannot go on PYTHONPATH", kernel_dir.display()))?;

        let mut child = Command::new(python)
            .arg("-m")
            .arg("corpus_kernel")
            // The kernel runs model-written code: it gets no inherited secrets, ever.
            .env_clear()
            .envs(machine_env())
            .env("PYTHONPATH", import_path)
            .env("PYTHONUNBUFFERED", "1")
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .env("PYTHONIOENCODING", "utf-8")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .context("failed to spawn python kernel")?;

        let writes = line_writer(child.stdin.take().expect("piped"));
        let stdout = child.stdout.take().expect("piped");
        let stderr = child.stderr.take().expect("piped");

        let (tx, frames) = mpsc::channel(256);
        tokio::spawn(async move {
            let mut lines = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if let Ok(frame) = serde_json::from_str::<Frame>(&line)
                    && tx.send(frame).await.is_err()
                {
                    break;
                }
            }
        });

        let log = Arc::new(Mutex::new(String::new()));
        let sink = log.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let mut buf = sink.lock().unwrap();
                buf.push_str(&line);
                buf.push('\n');
                if buf.len() > STDERR_TAIL {
                    let cut = buf.len() - STDERR_TAIL;
                    buf.drain(..cut);
                }
            }
        });

        writes
            .send(encode(&json!({ "type": "init", "fns": fns })))
            .context("kernel did not accept init")?;

        let mut kernel = Kernel {
            child,
            writes,
            frames,
            stderr: log,
        };
        match kernel.frames.recv().await {
            Some(Frame::Ready) => Ok(kernel),
            _ => bail!("kernel failed to start: {}", kernel.stderr_tail()),
        }
    }

    pub fn interrupter(&self) -> Interrupter {
        Interrupter(self.writes.clone())
    }

    /// Runs one cell. Host calls it makes are served on tasks of their own, so several can
    /// be in flight at once and none of them blocks the frames the cell is still sending.
    /// The tasks belong to this call: when it returns — finished, interrupted or killed —
    /// they are dropped, and an answer bought for a cell that is no longer there is never
    /// written back.
    pub async fn exec(
        &mut self,
        code: &str,
        host: &Arc<dyn Host>,
        on_stream: &mut (dyn FnMut(&str) + Send),
        timeout: Duration,
    ) -> Result<ExecOutcome> {
        let id = uuid::Uuid::now_v7().to_string();
        self.send(&json!({ "type": "exec", "id": id, "code": code }))?;

        let mut calls = tokio::task::JoinSet::new();
        // How many of this cell's own calls are waiting on the host. The clock measures the
        // cell's work, not the host's: a host call carries its own budget, and a cell that
        // spent an hour waiting on three fetches was never the runaway this timeout is for.
        let mut open = 0usize;
        let mut deadline = Instant::now() + timeout;
        let mut interrupted = false;
        loop {
            let frame = tokio::select! {
                biased;
                // Once the interrupt has gone out the grace period stands whatever is in
                // flight, or a cell could buy itself another lifetime by asking for a page.
                _ = tokio::time::sleep_until(deadline), if open == 0 || interrupted => {
                    if interrupted {
                        let _ = self.child.kill().await;
                        bail!("kernel ignored the interrupt and was killed; restart it to continue");
                    }
                    interrupted = true;
                    deadline = Instant::now() + KILL_GRACE;
                    self.send(&interrupt())?;
                    continue;
                }
                Some(served) = calls.join_next() => {
                    self.send(&served.context("a host call panicked")?)?;
                    open -= 1;
                    if open == 0 && !interrupted {
                        deadline = Instant::now() + timeout;
                    }
                    continue;
                }
                frame = self.frames.recv() => match frame {
                    Some(frame) => frame,
                    None => bail!("kernel died: {}", self.stderr_tail()),
                },
            };
            match frame {
                Frame::Stream { id: cell, text } if cell == id => on_stream(&text),
                Frame::HostRequest {
                    req_id,
                    cell,
                    func,
                    args,
                } => {
                    // A thread an abandoned cell left running keeps asking, and its
                    // requests are still answered — but only the running cell's own calls
                    // may buy it more clock.
                    let mine = cell == id;
                    let host = host.clone();
                    let serve = async move {
                        match host.call(&func, args).await {
                            Ok(value) => {
                                json!({ "type": "host_reply", "req_id": req_id, "ok": true, "value": value })
                            }
                            Err(error) => {
                                json!({ "type": "host_reply", "req_id": req_id, "ok": false, "error": error })
                            }
                        }
                    };
                    match mine {
                        true => {
                            open += 1;
                            calls.spawn(serve);
                        }
                        // Not the cell's, so not the cell's to abandon: a thread waiting
                        // on this has no interrupt that can reach it, and an answer that
                        // never comes parks it for as long as the kernel lives. It writes
                        // its own reply, because this loop will not be here for it.
                        false => {
                            let writes = self.writes.clone();
                            tokio::spawn(async move {
                                let _ = writes.send(encode(&serve.await));
                            });
                        }
                    }
                }
                Frame::Done {
                    id: cell,
                    status,
                    repr,
                    traceback,
                } if cell == id => {
                    return Ok(ExecOutcome {
                        ok: status == "ok",
                        repr,
                        traceback,
                    });
                }
                _ => {} // frames from an abandoned cell
            }
        }
    }

    fn stderr_tail(&self) -> String {
        let tail = self.stderr.lock().unwrap().trim().to_string();
        if tail.is_empty() {
            "no output on stderr".into()
        } else {
            tail
        }
    }

    fn send(&self, frame: &Value) -> Result<()> {
        if self.writes.send(encode(frame)).is_err() {
            bail!("kernel died: {}", self.stderr_tail());
        }
        Ok(())
    }
}

use std::ffi::OsStr;
use std::path::Path;
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use serde::Deserialize;
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

/// Reaches a running cell from outside the `exec` that is awaiting it.
#[derive(Clone)]
pub struct Interrupter(mpsc::UnboundedSender<String>);

impl Interrupter {
    pub fn raise(&self) {
        let _ = self.0.send(encode(&interrupt()));
    }
}

/// A frame as it goes on the wire: one JSON object, one line.
fn encode(frame: &Value) -> String {
    format!("{frame}\n")
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
    pub async fn start(
        python: impl AsRef<OsStr>,
        kernel_dir: &Path,
        fns: &[&str],
    ) -> Result<Kernel> {
        let mut child = Command::new(python)
            .arg("-m")
            .arg("corpus_kernel")
            // The kernel runs model-written code: it gets no inherited secrets, ever.
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("PYTHONPATH", kernel_dir)
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

    pub async fn exec(
        &mut self,
        code: &str,
        host: &dyn Host,
        on_stream: &mut (dyn FnMut(&str) + Send),
        timeout: Duration,
    ) -> Result<ExecOutcome> {
        let id = uuid::Uuid::now_v7().to_string();
        self.send(&json!({ "type": "exec", "id": id, "code": code }))?;

        let mut deadline = Instant::now() + timeout;
        let mut interrupted = false;
        loop {
            let frame = match tokio::time::timeout_at(deadline, self.frames.recv()).await {
                Err(_) if !interrupted => {
                    interrupted = true;
                    deadline = Instant::now() + KILL_GRACE;
                    self.send(&interrupt())?;
                    continue;
                }
                Err(_) => {
                    let _ = self.child.kill().await;
                    bail!("kernel ignored the interrupt and was killed; restart it to continue");
                }
                Ok(None) => bail!("kernel died: {}", self.stderr_tail()),
                Ok(Some(frame)) => frame,
            };
            match frame {
                Frame::Stream { id: cell, text } if cell == id => on_stream(&text),
                Frame::HostRequest { req_id, func, args } => {
                    let reply = match host.call(&func, args).await {
                        Ok(value) => {
                            json!({ "type": "host_reply", "req_id": req_id, "ok": true, "value": value })
                        }
                        Err(error) => {
                            json!({ "type": "host_reply", "req_id": req_id, "ok": false, "error": error })
                        }
                    };
                    self.send(&reply)?;
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

use std::process::Stdio;

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use corpus_agent::{Agent, Event, Interrupt};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, BufReader, Lines};
use tokio::process::{Child, ChildStdout};
use tokio::sync::mpsc;
use uuid::Uuid;

/// One turn of conversation, wherever the loop actually runs. The renderer holds this
/// and cannot tell a local agent from one behind a pipe — that is the whole point (§8 Ш5).
#[async_trait]
pub trait Session: Send {
    async fn run(&mut self, prompt: &str, on_event: &mut (dyn FnMut(Event) + Send)) -> Result<()>;

    async fn finish(&mut self, on_event: &mut (dyn FnMut(Event) + Send)) -> Result<()>;

    /// Taken before the session starts running, and usable while it does.
    fn interrupt(&self) -> Interrupt;
}

/// What `corpus connect` writes to `corpus serve`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
pub enum Command {
    Run { text: String },
    Interrupt,
    Exit,
}

pub struct Local {
    pub session_id: Uuid,
    model: String,
    agent: Agent,
    announced: bool,
}

impl Local {
    pub fn new(agent: Agent, model: String) -> Local {
        Local {
            session_id: Uuid::now_v7(),
            model,
            agent,
            announced: false,
        }
    }
}

#[async_trait]
impl Session for Local {
    async fn run(&mut self, prompt: &str, on_event: &mut (dyn FnMut(Event) + Send)) -> Result<()> {
        if !self.announced {
            self.announced = true;
            on_event(Event::SessionStart {
                session_id: self.session_id,
                model: self.model.clone(),
            });
        }
        self.agent.run(prompt, on_event).await?;
        Ok(())
    }

    async fn finish(&mut self, on_event: &mut (dyn FnMut(Event) + Send)) -> Result<()> {
        on_event(Event::SessionEnd {
            session_id: self.session_id,
        });
        Ok(())
    }

    fn interrupt(&self) -> Interrupt {
        self.agent.interrupter()
    }
}

/// The same session, on the other side of a pipe. No crypto and no framing beyond
/// JSON lines: if this renders identically to [`Local`], the hosted mode can be added
/// later without touching the loop (§1).
pub struct Remote {
    child: Child,
    writes: mpsc::UnboundedSender<String>,
    lines: Lines<BufReader<ChildStdout>>,
}

impl Remote {
    pub async fn spawn(argv: &[String]) -> Result<Remote> {
        let (program, rest) = argv
            .split_first()
            .context("connect needs the command to run after `--`, e.g. `-- corpus serve`")?;
        let mut child = tokio::process::Command::new(program)
            .args(rest)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("could not start `{program}`"))?;
        let writes = corpus_kernel::line_writer(child.stdin.take().expect("piped"));
        let stdout = child.stdout.take().expect("piped");
        Ok(Remote {
            child,
            writes,
            lines: BufReader::new(stdout).lines(),
        })
    }

    fn send(&self, command: &Command) -> Result<()> {
        self.writes.send(encode(command))?;
        Ok(())
    }

    async fn pump(
        &mut self,
        on_event: &mut (dyn FnMut(Event) + Send),
        until: fn(&Event) -> bool,
    ) -> Result<()> {
        while let Some(line) = self.lines.next_line().await? {
            if line.trim().is_empty() {
                continue;
            }
            let event: Event = serde_json::from_str(&line)
                .with_context(|| format!("served an unreadable event: {line}"))?;
            let done = until(&event);
            on_event(event);
            if done {
                return Ok(());
            }
        }
        bail!("the served session ended without finishing the turn");
    }
}

#[async_trait]
impl Session for Remote {
    async fn run(&mut self, prompt: &str, on_event: &mut (dyn FnMut(Event) + Send)) -> Result<()> {
        self.send(&Command::Run {
            text: prompt.to_string(),
        })?;
        self.pump(on_event, |event| matches!(event, Event::TurnEnd { .. }))
            .await
    }

    async fn finish(&mut self, on_event: &mut (dyn FnMut(Event) + Send)) -> Result<()> {
        self.send(&Command::Exit)?;
        self.pump(on_event, |event| matches!(event, Event::SessionEnd { .. }))
            .await?;
        let _ = self.child.wait().await;
        Ok(())
    }

    fn interrupt(&self) -> Interrupt {
        let writes = self.writes.clone();
        let frame = encode(&Command::Interrupt);
        Interrupt::new(move || {
            let _ = writes.send(frame.clone());
        })
    }
}

/// A command as it goes on the wire: one JSON object, one line.
fn encode(command: &Command) -> String {
    format!(
        "{}\n",
        serde_json::to_string(command).expect("a string and a unit variant")
    )
}

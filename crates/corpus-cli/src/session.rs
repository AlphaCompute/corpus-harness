use std::process::Stdio;

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use corpus_agent::{Agent, Event, Interrupt};
use corpus_kernel::encode;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, BufReader, Lines};
use tokio::process::{Child, ChildStdout};
use tokio::sync::mpsc;
use uuid::Uuid;

/// One turn of conversation, wherever the loop actually runs. The renderer holds this
/// and cannot tell a local agent from one behind a pipe — that is the whole point.
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
    /// Taken by the announcement, which is what makes it happen once.
    model: Option<String>,
    agent: Agent,
    /// Everything the children say. It is drained alongside the turn rather than by it,
    /// because a child talks while its parent is thinking and neither waits on the other.
    children: mpsc::UnboundedReceiver<Event>,
}

impl Local {
    pub fn new(agent: Agent, model: String, children: mpsc::UnboundedReceiver<Event>) -> Local {
        Local {
            session_id: Uuid::now_v7(),
            model: Some(model),
            agent,
            children,
        }
    }
}

#[async_trait]
impl Session for Local {
    async fn run(&mut self, prompt: &str, on_event: &mut (dyn FnMut(Event) + Send)) -> Result<()> {
        if let Some(model) = self.model.take() {
            on_event(Event::SessionStart {
                session_id: self.session_id,
                model,
            });
        }
        // The turn's own events go down a channel rather than into `on_event`, so that
        // one caller is left holding it: the arms below. Sharing it with the children's
        // stream is what the borrow checker will not have, and what the alternative —
        // draining the children only when the parent happens to say something — gets
        // wrong anyway, because a parent waiting on a child says nothing for minutes.
        let Local {
            agent, children, ..
        } = self;
        let (told, mut mine) = mpsc::unbounded_channel();
        let mut sending = move |event| {
            let _ = told.send(event);
        };
        let turn = agent.run(prompt, &mut sending);
        tokio::pin!(turn);
        let outcome = loop {
            tokio::select! {
                biased;
                outcome = &mut turn => break outcome,
                Some(event) = mine.recv() => on_event(event),
                Some(event) = children.recv() => on_event(event),
            }
        };
        // The turn ends holding whatever it said last, `TurnEnd` among it: a caller that
        // learns the turn is over before it hears the end of it has been told twice.
        while let Ok(event) = mine.try_recv() {
            on_event(event);
        }
        while let Ok(event) = children.try_recv() {
            on_event(event);
        }
        outcome?;
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
/// later without touching the loop.
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

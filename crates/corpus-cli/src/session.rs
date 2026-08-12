use std::process::Stdio;

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use corpus_agent::{Agent, Event, Interrupt};
use corpus_kernel::encode;
use corpus_provider::Usage;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, BufReader, Lines};
use tokio::process::{Child, ChildStdout};
use tokio::sync::mpsc;
use uuid::Uuid;

/// What a turn starts from: a person, or the children reporting back. Whoever drives a
/// session hands one of these back rather than running a turn from inside a `select!`,
/// where the arm that produced it is still holding the renderer.
pub enum Prompt {
    Human(String),
    Notice(String),
}

/// One turn of conversation, wherever the loop actually runs. The renderer holds this
/// and cannot tell a local agent from one behind a pipe — that is the whole point.
#[async_trait]
pub trait Session: Send {
    async fn run(&mut self, prompt: &str, on_event: &mut (dyn FnMut(Event) + Send)) -> Result<()>;

    /// A turn started by what the children reported rather than by a person.
    async fn notified(
        &mut self,
        news: &str,
        on_event: &mut (dyn FnMut(Event) + Send),
    ) -> Result<()>;

    /// What the session does while nobody is typing. It renders whatever happens on its
    /// own — children start, work and come back between turns — and returns the news that
    /// is worth a turn of its own, or nothing when there is none to be had. Never returns
    /// on an idle session, and safe to drop the moment a person types.
    async fn idle(&mut self, on_event: &mut (dyn FnMut(Event) + Send)) -> Result<Option<String>>;

    async fn finish(&mut self, on_event: &mut (dyn FnMut(Event) + Send)) -> Result<()>;

    /// Taken before the session starts running, and usable while it does.
    fn interrupt(&self) -> Interrupt;
}

/// Kept for the boundary. A free function because a turn has the session taken apart
/// field by field, which is what lets a child be heard while its parent is thinking.
fn keep(news: &mut Vec<Event>, spent: &mut Usage, event: &Event) {
    match event {
        Event::AgentEnd { .. } => news.push(event.clone()),
        // Only a child's turns come this way, and what a child spent is spent by the
        // session: it goes on the turn's bill. Not on `context` — that one says how full
        // the window is, and a child fills a window of its own.
        Event::TurnEnd { usage, .. } => {
            spent.input += usage.input;
            spent.output += usage.output;
        }
        _ => {}
    }
}

/// The turn's own ending, with whatever its children spent while it ran folded in.
fn bill(spent: &mut Usage, event: &mut Event) {
    if let Event::TurnEnd { usage, .. } = event {
        usage.input += spent.input;
        usage.output += spent.output;
        *spent = Usage::default();
    }
}

/// Runs whichever kind of turn this is. Every driver has both cases and none of them
/// needs to know how they differ.
pub async fn deliver(
    session: &mut dyn Session,
    prompt: Prompt,
    on_event: &mut (dyn FnMut(Event) + Send),
) -> Result<()> {
    match prompt {
        Prompt::Human(text) => session.run(&text, on_event).await,
        Prompt::Notice(text) => session.notified(&text, on_event).await,
    }
}

/// How many turns a session may take on what its children said before it waits to be
/// spoken to again. Each one is a whole turn under its own budget, so the ceiling is not
/// about cost but about a session that answers itself: children reporting to children.
/// Only a person typing resets it.
const AUTONOMOUS: u32 = 8;

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
    /// Turns taken since a person last said anything.
    autonomous: u32,
    /// Children that came back and have not been reported yet. A child finishing while
    /// its parent was busy is exactly the case notifying exists for, so what lands mid-turn
    /// waits here for the boundary rather than being shown and forgotten.
    news: Vec<Event>,
    /// What the children have spent and nobody has been billed for yet.
    spent: Usage,
}

impl Local {
    pub fn new(agent: Agent, model: String, children: mpsc::UnboundedReceiver<Event>) -> Local {
        Local {
            session_id: Uuid::now_v7(),
            model: Some(model),
            agent,
            children,
            autonomous: 0,
            news: Vec::new(),
            spent: Usage::default(),
        }
    }
}

#[async_trait]
impl Session for Local {
    async fn run(&mut self, prompt: &str, on_event: &mut (dyn FnMut(Event) + Send)) -> Result<()> {
        self.autonomous = 0;
        self.turn(prompt, false, on_event).await
    }

    async fn notified(
        &mut self,
        news: &str,
        on_event: &mut (dyn FnMut(Event) + Send),
    ) -> Result<()> {
        self.turn(news, true, on_event).await
    }

    /// Waits for a child to come back, showing whatever else happens on the way, and folds
    /// everything that has landed since the last boundary into one telling: eight children
    /// finishing must be one turn, not eight.
    async fn idle(&mut self, on_event: &mut (dyn FnMut(Event) + Send)) -> Result<Option<String>> {
        loop {
            // At the ceiling this simply never comes true: the session goes on showing
            // what its children do and waits to be spoken to. Nothing is lost — the
            // answers are where they always were, in the agents that hold them.
            if !self.news.is_empty() && self.autonomous < AUTONOMOUS {
                self.autonomous += 1;
                return Ok(crate::children::doorbell(&std::mem::take(&mut self.news)));
            }
            let Some(event) = self.children.recv().await else {
                // Nobody is left to say anything, which is not an answer this can give.
                return std::future::pending().await;
            };
            self.remember(&event);
            on_event(event);
            while let Ok(event) = self.children.try_recv() {
                self.remember(&event);
                on_event(event);
            }
        }
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

impl Local {
    fn remember(&mut self, event: &Event) {
        keep(&mut self.news, &mut self.spent, event);
    }

    async fn turn(
        &mut self,
        prompt: &str,
        from_children: bool,
        on_event: &mut (dyn FnMut(Event) + Send),
    ) -> Result<()> {
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
            agent,
            children,
            news,
            spent,
            ..
        } = self;
        let (told, mut mine) = mpsc::unbounded_channel();
        let mut sending = move |event| {
            let _ = told.send(event);
        };
        let turn = agent.turn(prompt, from_children, &mut sending);
        tokio::pin!(turn);
        let outcome = loop {
            tokio::select! {
                biased;
                outcome = &mut turn => break outcome,
                Some(mut event) = mine.recv() => {
                    bill(spent, &mut event);
                    on_event(event);
                }
                Some(event) = children.recv() => {
                    keep(news, spent, &event);
                    on_event(event);
                }
            }
        };
        // The turn ends holding whatever it said last, `TurnEnd` among it: a caller that
        // learns the turn is over before it hears the end of it has been told twice.
        while let Ok(mut event) = mine.try_recv() {
            bill(spent, &mut event);
            on_event(event);
        }
        while let Ok(event) = children.try_recv() {
            keep(news, spent, &event);
            on_event(event);
        }
        outcome?;
        Ok(())
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

    /// The children live where the loop does, so a turn started by their news is started
    /// over there. This side only has to keep reading: an autonomous turn nobody read for
    /// would be invisible here until somebody typed.
    async fn notified(&mut self, _: &str, _: &mut (dyn FnMut(Event) + Send)) -> Result<()> {
        bail!("a served session starts its own turns")
    }

    async fn idle(&mut self, on_event: &mut (dyn FnMut(Event) + Send)) -> Result<Option<String>> {
        self.pump(on_event, |_| false).await?;
        std::future::pending().await
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

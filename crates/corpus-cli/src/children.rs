//! Agents the model sends off with a task of their own. A child is the same agent this
//! session is: its own loop, its own interpreter, the same functions in its namespace. It
//! is reached from the cell that made it, by a handle that is nothing but an id.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use corpus_agent::{Agent, Budget, Event, Kernels};
use corpus_kernel::Kernel;
use corpus_provider::Provider;
use serde_json::{Value, json};
use tokio::sync::{Semaphore, mpsc, watch};
use uuid::Uuid;

use crate::host::{Search, Tools, ellipsize, fence};

/// How many children may be taking a turn at once. The rest are not refused — a refusal
/// would be one more thing for the model to handle — they wait their turn, so a hundred
/// children spawned in one cell are a queue and not a hundred inference calls at once.
const AT_ONCE: usize = 8;

/// What `result()` waits when it was not told, and the most it will wait when it was. The
/// point of the ceiling is not the clock — a cell waiting on the host spends none of its
/// own budget — but the shape of the work: a loop over short waits leaves the model able
/// to do something else, and leaves a person able to interrupt into a cell that answers.
const WAIT: Duration = Duration::from_secs(30);
const LONGEST_WAIT: Duration = Duration::from_secs(300);

/// A doorbell, not the delivery: enough to tell which child came back and whether it is
/// worth reading now. The whole of it is one `result()` away.
const PREVIEW: usize = 160;

/// Everything it takes to build an agent away from the place the session was built.
#[derive(Clone)]
pub struct Recipe {
    pub python: String,
    pub kernel_dir: PathBuf,
    /// The skills roots the session was started with, so a child imports the same packages
    /// its parent's prompt describes rather than whatever the directory it ends up in has.
    pub roots: Vec<PathBuf>,
    pub provider: Provider,
    pub budget: Budget,
    pub search: Option<Search>,
    /// What the prompt says about the skills, as it was rendered for the session: a child
    /// starts the same kernel off the same directory, so it sees the same ones.
    pub skills: String,
}

/// Where a child is between turns. `running` is not a field: it is `busy || queued > 0`,
/// which is what keeps a child that was made a microsecond ago from reading as finished.
#[derive(Clone, Default)]
struct Status {
    busy: bool,
    queued: usize,
    /// How the last finished turn went, or nothing at all if none has finished. One field
    /// rather than two, so an answer and a failure cannot both be on record at once.
    last: Option<Result<String, String>>,
}

impl Status {
    fn settled(&self) -> bool {
        !self.busy && self.queued == 0
    }
}

struct Kid {
    task: String,
    inbox: mpsc::UnboundedSender<String>,
    /// Held as the sender, because both sides move it: the host counts what it queues,
    /// the child's own task counts what it has taken and answered.
    status: Arc<watch::Sender<Status>>,
}

pub struct Children {
    recipe: Recipe,
    parent: Uuid,
    /// Everything the children say, on its way to the session's log and screen.
    events: mpsc::UnboundedSender<Event>,
    turns: Arc<Semaphore>,
    kids: Mutex<HashMap<Uuid, Kid>>,
}

impl Children {
    pub const NAMES: [&'static str; 5] = ["spawn", "result", "send", "done", "agents"];

    pub fn new(recipe: Recipe, parent: Uuid, events: mpsc::UnboundedSender<Event>) -> Children {
        Children {
            recipe,
            parent,
            events,
            turns: Arc::new(Semaphore::new(AT_ONCE)),
            kids: Mutex::new(HashMap::new()),
        }
    }

    /// Whose session this is. Everything a host function says happens is said in this
    /// agent's name, because it is the one whose cell asked for it.
    pub fn parent(&self) -> Uuid {
        self.parent
    }

    /// One channel carries everything that happens away from the turn's own loop, and the
    /// children are only its first users. Nothing waits on the far end: a session that
    /// stopped listening is a session that is over.
    pub fn announce(&self, event: Event) {
        let _ = self.events.send(event);
    }

    pub async fn call(&self, name: &str, args: &Value) -> Result<Value, String> {
        match name {
            "spawn" => Ok(self.spawn(text(args, "task")?)),
            "result" => self.result(self.named(args)?, wait(args)).await,
            "send" => self.send(self.named(args)?, text(args, "text")?),
            "done" => self.done(self.named(args)?),
            "agents" => Ok(self.agents()),
            other => Err(format!("no host function named `{other}`")),
        }
    }

    /// Returns as soon as the child exists, which is before it has started: an agent's
    /// interpreter takes a moment to come up, and no cell should wait on that to get a
    /// handle back. The task is in the inbox, so the child is busy from this instant.
    fn spawn(&self, task: &str) -> Value {
        let id = Uuid::now_v7();
        let (inbox, mail) = mpsc::unbounded_channel();
        let (status, _) = watch::channel(Status {
            queued: 1,
            ..Status::default()
        });
        let status = Arc::new(status);
        let _ = inbox.send(task.to_string());
        self.kids.lock().unwrap().insert(
            id,
            Kid {
                task: task.to_string(),
                inbox,
                status: status.clone(),
            },
        );
        let _ = self.events.send(Event::AgentStart {
            agent: id,
            parent: self.parent,
            task: task.to_string(),
        });
        self.raise(id, mail, status);
        json!(id.to_string())
    }

    /// The child's own loop: one turn per message, for as long as anyone can reach it.
    /// It ends when the session does, because the inbox is only held by the registry.
    fn raise(
        &self,
        id: Uuid,
        mut mail: mpsc::UnboundedReceiver<String>,
        status: Arc<watch::Sender<Status>>,
    ) {
        let recipe = self.recipe.clone();
        let events = self.events.clone();
        let turns = self.turns.clone();
        let parent = self.parent;
        tokio::spawn(async move {
            let mut agent = leaf(&recipe, id);
            while let Some(text) = mail.recv().await {
                let permit = turns.acquire().await;
                status.send_modify(|state| {
                    state.busy = true;
                    state.queued = state.queued.saturating_sub(1);
                });
                let outcome = agent
                    .run(&text, &mut |event| {
                        let _ = events.send(event);
                    })
                    .await;
                let answered = match outcome {
                    Ok(outcome) => Ok(outcome.text),
                    // A child that broke is data for whoever asked, the way a cell that
                    // raised is: the parent hears about it and decides what to do.
                    Err(failure) => Err(format!("{failure:#}")),
                };
                let _ = events.send(Event::AgentEnd {
                    agent: id,
                    parent,
                    ok: answered.is_ok(),
                    chars: answered.as_ref().map_or(0, |text| text.chars().count()),
                    preview: ellipsize(
                        match &answered {
                            Ok(text) => text,
                            Err(why) => why,
                        },
                        PREVIEW,
                    ),
                });
                // The latest turn is the answer, whichever way it went: an agent that
                // worked once and broke since must not keep handing back what it said
                // the first time.
                status.send_modify(|state| {
                    state.busy = false;
                    state.last = Some(answered);
                });
                drop(permit);
            }
        });
    }

    /// What the last finished turn answered, or nothing at all if it has not finished.
    /// Never the block that would make a for-loop over several children impossible.
    async fn result(&self, id: Uuid, wait: Duration) -> Result<Value, String> {
        let mut status = self.watching(id)?;
        let settled = tokio::time::timeout(wait, status.wait_for(Status::settled)).await;
        let last = match settled {
            Err(_) => return Ok(Value::Null),
            Ok(Err(_)) => return Err(format!("agent {id} is gone")),
            Ok(Ok(state)) => state.last.clone(),
        };
        match last {
            // A child reports on material it went and read, so what it says is material
            // too: the same fence a fetched page gets, for the same reason.
            Some(Ok(answer)) => Ok(json!(fence(&format!("agent {id}"), &answer))),
            Some(Err(why)) => Err(why),
            None => Ok(Value::Null),
        }
    }

    /// Another turn for a child that already exists. Busy is not a reason to refuse: the
    /// inbox is the queue, and the child takes it when it comes back up for air.
    fn send(&self, id: Uuid, text: &str) -> Result<Value, String> {
        let kids = self.kids.lock().unwrap();
        let kid = kids.get(&id).ok_or(format!("no agent {id}"))?;
        kid.status.send_modify(|state| state.queued += 1);
        kid.inbox
            .send(text.to_string())
            .map_err(|_| format!("agent {id} is gone"))?;
        Ok(json!(true))
    }

    fn done(&self, id: Uuid) -> Result<Value, String> {
        let kids = self.kids.lock().unwrap();
        let kid = kids.get(&id).ok_or(format!("no agent {id}"))?;
        let settled = kid.status.borrow().settled();
        Ok(json!(settled))
    }

    /// Every child this session has, in the order they were made.
    fn agents(&self) -> Value {
        let kids = self.kids.lock().unwrap();
        let mut ids: Vec<&Uuid> = kids.keys().collect();
        ids.sort();
        json!(
            ids.iter()
                .map(|id| json!({ "agent": id.to_string(), "task": kids[id].task }))
                .collect::<Vec<Value>>()
        )
    }

    /// A receiver of its own, taken out from under the lock: waiting on one while holding
    /// the registry would stop every other child from being reached.
    fn watching(&self, id: Uuid) -> Result<watch::Receiver<Status>, String> {
        let kids = self.kids.lock().unwrap();
        let kid = kids.get(&id).ok_or(format!("no agent {id}"))?;
        Ok(kid.status.subscribe())
    }

    fn named(&self, args: &Value) -> Result<Uuid, String> {
        let id = text(args, "agent")?;
        Uuid::parse_str(id).map_err(|_| format!("`{id}` is not an agent"))
    }
}

/// A child of a child is not something this builds: the depth is in the list of names its
/// namespace is bound from, so a leaf is simply never handed `spawn`.
fn leaf(recipe: &Recipe, id: Uuid) -> Agent {
    let python = recipe.python.clone();
    let kernel_dir = recipe.kernel_dir.clone();
    let roots = recipe.roots.clone();
    let start = Kernels::new(move || {
        let python = python.clone();
        let kernel_dir = kernel_dir.clone();
        let roots = roots.clone();
        async move { Kernel::start(&python, &kernel_dir, &roots, &Tools::names(false)).await }
    });
    Agent::lazy(
        id,
        recipe.provider.clone(),
        start,
        Arc::new(Tools::new(
            recipe.provider.clone(),
            recipe.search.clone(),
            None,
        )),
        &crate::system_prompt(Some(id), &recipe.skills),
        recipe.budget,
    )
}

/// What the children finishing is worth saying to whoever sent them: which one, how it
/// went, and enough to decide whether to go and read the rest. Not the rest itself — an
/// answer pasted here would sit in the context forever, where compaction cannot reach a
/// user message, and it is one `result()` away in any case.
pub fn doorbell(news: &[Event]) -> Option<String> {
    let lines: Vec<String> = news
        .iter()
        .filter_map(|event| match event {
            Event::AgentEnd {
                agent,
                ok,
                chars,
                preview,
                ..
            } => Some(format!(
                "agent {agent} {} · {chars} chars · {preview}",
                match ok {
                    true => "finished",
                    false => "failed",
                }
            )),
            _ => None,
        })
        .collect();
    match lines.is_empty() {
        true => None,
        // Fenced, because the openings quoted in here are the agents' own words, and an
        // agent that read a page is repeating what the page said. This is the one place a
        // child's text reaches the parent without passing through `result`.
        // Where the rest of an answer is kept is in the prompt, said once for the session
        // rather than again on every child that comes back.
        false => Some(fence("your agents", &lines.join("\n"))),
    }
}

fn text<'a>(args: &'a Value, field: &str) -> Result<&'a str, String> {
    args[field]
        .as_str()
        .ok_or(format!("this call needs `{field}` as a string"))
}

fn wait(args: &Value) -> Duration {
    args["timeout"]
        .as_f64()
        .filter(|seconds| *seconds > 0.0)
        .map_or(WAIT, |seconds| {
            Duration::from_secs_f64(seconds).min(LONGEST_WAIT)
        })
}

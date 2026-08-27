use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use corpus_kernel::{Binding, Host, Kernel};
use corpus_provider::{Delta, Message, Provider, Role, StopReason, Tool, ToolCall, Usage};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use uuid::Uuid;

const TOOL_RESULT_LIMIT: usize = 8000;
const DROPPED: &str =
    "[output dropped to save context; whatever the cell bound is still alive in the session]";

/// A replayed transcript is full of promises about `_` and about variables that the log
/// cannot restore, so the model is told once, up front, that only the dialogue came back.
const RESUMED: &str = "\n\nThis session was resumed from a log. The interpreter is fresh: \
    the variables, imports and `_` the transcript talks about no longer exist.";

/// Where a turn's context went, in tokens. The provider hands back the prompt's exact
/// total and says nothing about its parts, so the parts are apportioned by serialized
/// size: near enough to answer "where did my window go", and never exact. Nothing shows
/// these three without marking them as approximate, and nothing presents them as summing
/// to the total they were derived from.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextShape {
    pub system: u32,
    pub tools: u32,
    pub messages: u32,
}

/// The session log. The model's context is derived from it, never stored beside it,
/// which is what lets compaction drop tool output without losing the session.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "t", rename_all = "snake_case")]
pub enum Event {
    SessionStart {
        session_id: Uuid,
        model: String,
        /// How much context the model was advertised to have. Zero is a provider that
        /// never said, and a reading with no denominator is not shown at all.
        #[serde(default)]
        window: u32,
    },
    TurnStart {
        turn_id: Uuid,
        agent: Uuid,
    },
    UserMessage {
        turn_id: Uuid,
        /// Who was asked. Absent from logs written before agents had children, and a log
        /// that never names one is a log with only a root in it.
        #[serde(default)]
        agent: Uuid,
        text: String,
    },
    /// A turn nobody typed: what the children have to say, delivered as its prompt. Kept
    /// apart from a `UserMessage` so a replayed session cannot mistake it for a person.
    Notice {
        turn_id: Uuid,
        agent: Uuid,
        text: String,
    },
    AgentStart {
        agent: Uuid,
        parent: Uuid,
        task: String,
    },
    AgentEnd {
        agent: Uuid,
        parent: Uuid,
        ok: bool,
        chars: usize,
        /// The opening of what came back — a doorbell, not the delivery. The whole of it
        /// is fetched from the agent in code, so this is what the parent is told and all
        /// that a log needs to rebuild the telling.
        preview: String,
    },
    ThinkingDelta {
        agent: Uuid,
        text: String,
    },
    MessageDelta {
        agent: Uuid,
        text: String,
    },
    CodeDelta {
        agent: Uuid,
        text: String,
    },
    ToolStart {
        agent: Uuid,
        call_id: String,
        name: String,
        args: Value,
    },
    ToolStream {
        agent: Uuid,
        call_id: String,
        text: String,
    },
    ToolEnd {
        agent: Uuid,
        call_id: String,
        ok: bool,
        summary: String,
        ms: u64,
    },
    /// A file the session is handing to whoever it is talking to. The bytes stay on disk:
    /// whoever delivers reads the path itself, so a log keeps the fact of the delivery and
    /// not a copy of the file.
    UserFile {
        agent: Uuid,
        path: String,
        filename: String,
        bytes: u64,
        caption: Option<String>,
    },
    Compaction {
        agent: Uuid,
        dropped: usize,
    },
    /// What a cell left in the namespace, and what it took away.
    ///
    /// Names and shapes only, never the values: the namespace is where a run's data lives
    /// precisely so that it does not have to travel, and a log that carried it would undo
    /// the one thing the design is for. What this answers is the question the log could
    /// not — what does the model have to work with in there.
    Namespace {
        agent: Uuid,
        call_id: String,
        names: Vec<Binding>,
        #[serde(default)]
        gone: Vec<String>,
        /// Names the summary would not fit, said rather than dropped in silence.
        #[serde(default)]
        trimmed: u32,
    },
    /// One pass through the loop: one model call and whatever tools it asked for. A step
    /// that answers in text calls no tools and so leaves no other trace, which is why the
    /// count cannot be recovered from the rest of the log.
    ///
    /// `usage` is this step's alone — a child's is never folded in. Summing steps is the
    /// only way to total a session without counting a delegated turn twice, because
    /// `TurnEnd.usage` deliberately carries the children's spend as well.
    StepEnd {
        turn_id: Uuid,
        agent: Uuid,
        step: u32,
        /// The whole model call, retries and their backoff included.
        llm_ms: u64,
        /// Request sent to the first delta a reader would have seen, on the attempt that
        /// answered. Absent when the model never said anything.
        #[serde(default)]
        ttft_ms: Option<u64>,
        usage: Usage,
    },
    TurnEnd {
        turn_id: Uuid,
        agent: Uuid,
        stop: StopReason,
        usage: Usage,
        /// Wall time from the prompt landing to the answer, which is not the sum of the
        /// steps' model time: tools run between them, and children run beside them.
        #[serde(default)]
        wall_ms: u64,
        /// Where `context` went, apportioned by serialized size. An approximation, and
        /// marked as one wherever it is shown.
        #[serde(default)]
        shape: ContextShape,
        /// What the last step of the turn sent, which is what the session is carrying
        /// now. `usage.input` is the turn's whole bill and counts a long turn's context
        /// once per step, so it says nothing about how full the window is.
        #[serde(default)]
        context: u32,
    },
    Answer {
        agent: Uuid,
        text: String,
    },
    SessionEnd {
        session_id: Uuid,
    },
}

impl Event {
    /// Whose transcript this belongs to, which for an event about a child is the parent
    /// it is news for: a child's own stream is the child's, the line saying it started is
    /// the parent's. `None` is the session itself, which belongs to everyone.
    pub fn agent(&self) -> Option<Uuid> {
        match self {
            Event::SessionStart { .. } | Event::SessionEnd { .. } => None,
            Event::AgentStart { parent, .. } | Event::AgentEnd { parent, .. } => Some(*parent),
            Event::TurnStart { agent, .. }
            | Event::UserMessage { agent, .. }
            | Event::Notice { agent, .. }
            | Event::ThinkingDelta { agent, .. }
            | Event::MessageDelta { agent, .. }
            | Event::CodeDelta { agent, .. }
            | Event::ToolStart { agent, .. }
            | Event::ToolStream { agent, .. }
            | Event::ToolEnd { agent, .. }
            | Event::UserFile { agent, .. }
            | Event::Compaction { agent, .. }
            | Event::Namespace { agent, .. }
            | Event::StepEnd { agent, .. }
            | Event::TurnEnd { agent, .. }
            | Event::Answer { agent, .. } => Some(*agent),
        }
    }
}

/// What the children have to say, as the model reads it. The wrapper is what keeps a
/// turn nobody typed from reading like a person, live and replayed alike.
pub fn news(text: &str) -> String {
    format!("[your subagents, not the person you are talking to]\n{text}")
}

/// A way to stop a turn that is already running, held by whoever is not holding the
/// session itself.
#[derive(Clone)]
pub struct Interrupt(Arc<dyn Fn() + Send + Sync>);

impl Interrupt {
    pub fn new(raise: impl Fn() + Send + Sync + 'static) -> Interrupt {
        Interrupt(Arc::new(raise))
    }

    pub fn raise(&self) {
        (self.0)()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Budget {
    pub max_steps: u32,
    pub wall_clock: Duration,
    pub cell_timeout: Duration,
    pub context_chars: usize,
}

impl Default for Budget {
    fn default() -> Budget {
        Budget {
            // A backstop against a loop spinning fast, not a budget for the work: at any
            // pace a real task keeps, the wall clock below is what stops it first. Set low
            // enough to bind, it cuts research off mid-stride and reads as a failure.
            max_steps: 200,
            wall_clock: Duration::from_secs(900),
            cell_timeout: Duration::from_secs(120),
            context_chars: 120_000,
        }
    }
}

/// The cell a `python` call carries. Whoever renders the call reads it through this
/// rather than spelling the tool's schema out a second time.
pub fn python_code(args: &Value) -> Option<&str> {
    args["code"].as_str()
}

pub struct Outcome {
    pub text: String,
    pub stop: StopReason,
}

/// How to start an interpreter, for an agent that may never ask for one. An agent that
/// only reads what it was sent and answers costs nothing but the model call.
pub struct Kernels(Box<dyn Fn() -> Starting + Send + Sync>);

type Starting = Pin<Box<dyn Future<Output = Result<Kernel>> + Send>>;

impl Kernels {
    pub fn new<F>(start: impl Fn() -> F + Send + Sync + 'static) -> Kernels
    where
        F: Future<Output = Result<Kernel>> + Send + 'static,
    {
        Kernels(Box::new(move || Box::pin(start())))
    }
}

pub struct Agent {
    pub id: Uuid,
    /// A monotonic counter rather than a flag: a receiver taken at the start of a turn
    /// only ever sees interrupts raised after it, so a stale one cannot kill the next turn.
    cancel: tokio::sync::watch::Sender<u64>,
    provider: Provider,
    kernel: Option<Kernel>,
    start: Option<Kernels>,
    /// Filled the moment a kernel exists, because whoever holds the interrupt took it
    /// before the first cell was ever written.
    stop: Arc<std::sync::Mutex<Option<corpus_kernel::Interrupter>>>,
    host: Arc<dyn Host>,
    messages: Vec<Message>,
    tools: Vec<Tool>,
    budget: Budget,
}

impl Agent {
    /// An agent whose interpreter is already running, which is worth the wait only where
    /// something else was being waited on anyway. The id comes from outside: whoever
    /// makes an agent has to be able to say whose events these are before it exists.
    pub fn new(
        id: Uuid,
        provider: Provider,
        kernel: Kernel,
        host: Arc<dyn Host>,
        system_prompt: &str,
        budget: Budget,
    ) -> Agent {
        Agent::build(
            id,
            provider,
            Some(kernel),
            None,
            host,
            system_prompt,
            budget,
        )
    }

    /// An agent that starts its interpreter the first time it writes a cell.
    pub fn lazy(
        id: Uuid,
        provider: Provider,
        start: Kernels,
        host: Arc<dyn Host>,
        system_prompt: &str,
        budget: Budget,
    ) -> Agent {
        Agent::build(id, provider, None, Some(start), host, system_prompt, budget)
    }

    fn build(
        id: Uuid,
        provider: Provider,
        kernel: Option<Kernel>,
        start: Option<Kernels>,
        host: Arc<dyn Host>,
        system_prompt: &str,
        budget: Budget,
    ) -> Agent {
        let stop = kernel.as_ref().map(Kernel::interrupter);
        Agent {
            id,
            cancel: tokio::sync::watch::channel(0).0,
            provider,
            kernel,
            start,
            stop: Arc::new(std::sync::Mutex::new(stop)),
            host,
            messages: vec![Message::text(Role::System, system_prompt)],
            tools: vec![Tool {
                name: "python".into(),
                description: "Run Python in a session that keeps its variables between calls. \
                    Everything else — search, fetching, files, delivery — is a function in this namespace."
                    .into(),
                parameters: json!({
                    "type": "object",
                    "properties": { "code": { "type": "string" } },
                    "required": ["code"],
                }),
            }],
            budget,
        }
    }

    /// Stops the turn: the running cell is interrupted and the loop gives back the best
    /// partial answer instead of starting another step.
    pub fn interrupter(&self) -> Interrupt {
        let cancel = self.cancel.clone();
        let stop = self.stop.clone();
        Interrupt::new(move || {
            cancel.send_modify(|count| *count += 1);
            // Nothing to interrupt is not a failure to interrupt: an agent that has not
            // written a cell yet is stopped by the counter above and nothing else.
            if let Some(kernel) = stop.lock().unwrap().as_ref() {
                kernel.raise();
            }
        })
    }

    /// The interpreter, started if this is the first cell to need one.
    async fn kernel(&mut self) -> Result<&mut Kernel> {
        if self.kernel.is_none() {
            let starting = self
                .start
                .as_ref()
                .map(|start| (start.0)())
                .context("this agent has no interpreter to start")?;
            let kernel = starting.await?;
            *self.stop.lock().unwrap() = Some(kernel.interrupter());
            self.kernel = Some(kernel);
        }
        Ok(self.kernel.as_mut().expect("just started"))
    }

    /// Rebuilds the model's context from the session log. The kernel namespace does not
    /// come back: variables from the earlier run are gone, which the system message says
    /// out loud so the transcript's own promises do not mislead the model.
    ///
    /// Only the root's own dialogue comes back. A root cannot be named outright — this
    /// process minted a fresh id, and the log's `session_start` never carried one — so it
    /// is whoever no `agent_start` ever announced. Read that way, a log that was resumed
    /// and resumed again, with a generation of roots in it, still replays as one thread.
    pub fn replay(&mut self, events: impl IntoIterator<Item = Event>) {
        if let Some(system) = self.messages.first_mut()
            && let Some(prompt) = system.content.as_mut()
        {
            prompt.push_str(RESUMED);
        }
        let events: Vec<Event> = events.into_iter().collect();
        let children: std::collections::HashSet<Uuid> = events
            .iter()
            .filter_map(|event| match event {
                Event::AgentStart { agent, .. } => Some(*agent),
                _ => None,
            })
            .collect();

        let mut pending: Option<ToolCall> = None;
        let mut output = String::new();
        for event in events {
            if event.agent().is_some_and(|agent| children.contains(&agent)) {
                continue;
            }
            match event {
                Event::UserMessage { text, .. } => {
                    self.messages.push(Message::text(Role::User, text))
                }
                Event::Notice { text, .. } => {
                    self.messages.push(Message::text(Role::User, news(&text)))
                }
                Event::Answer { text, .. } if !text.is_empty() => {
                    self.messages.push(Message::text(Role::Assistant, text))
                }
                Event::ToolStart {
                    call_id,
                    name,
                    args,
                    ..
                } => {
                    pending = Some(ToolCall {
                        id: call_id,
                        name,
                        arguments: args.to_string(),
                    });
                    output.clear();
                }
                Event::ToolStream { text, .. } => output.push_str(&text),
                Event::ToolEnd { ok, summary, .. } => {
                    if let Some(call) = pending.take() {
                        let id = call.id.clone();
                        self.messages
                            .push(Message::calls(String::new(), vec![call]));
                        // The log kept the streamed output; a cell whose output was only its
                        // repr left just the summary behind.
                        let body = match ok {
                            true if output.is_empty() => summary.clone(),
                            _ => std::mem::take(&mut output),
                        };
                        self.messages
                            .push(Message::tool_result(id, tool_result(ok, body, &summary)));
                    }
                }
                _ => {}
            }
        }
    }

    pub async fn run(
        &mut self,
        prompt: &str,
        on_event: &mut (dyn FnMut(Event) + Send),
    ) -> Result<Outcome> {
        self.turn(prompt, false, on_event).await
    }

    /// One turn, from a person or from the children — the difference is what the log
    /// records and what the model is told it is reading. Whoever drives a session picks
    /// which at run time, so the two cannot be separate methods: they would be two
    /// different futures to hold in one place.
    pub async fn turn(
        &mut self,
        prompt: &str,
        from_children: bool,
        on_event: &mut (dyn FnMut(Event) + Send),
    ) -> Result<Outcome> {
        let turn_id = Uuid::now_v7();
        on_event(Event::TurnStart {
            turn_id,
            agent: self.id,
        });
        let said = match from_children {
            true => news(prompt),
            false => prompt.to_string(),
        };
        on_event(match from_children {
            true => Event::Notice {
                turn_id,
                agent: self.id,
                text: prompt.to_string(),
            },
            false => Event::UserMessage {
                turn_id,
                agent: self.id,
                text: prompt.to_string(),
            },
        });
        self.messages.push(Message::text(Role::User, said));

        let started = Instant::now();
        let mut cancelled = self.cancel.subscribe();
        let mut usage = Usage::default();
        let mut context = 0;
        let mut answer = String::new();
        let mut steps = 0;
        // Taken while `self.messages` still holds exactly what was measured. By the time
        // the turn ends the answer and its tool results have been pushed, and apportioning
        // the measured total across those would credit the conversation with bytes the
        // provider never counted.
        let mut shape = ContextShape::default();

        let stop = loop {
            if cancelled.has_changed().unwrap_or(false) {
                break StopReason::Partial;
            }
            if steps >= self.budget.max_steps || started.elapsed() >= self.budget.wall_clock {
                break StopReason::Partial;
            }
            steps += 1;

            let dropped = self.compact();
            if dropped > 0 {
                on_event(Event::Compaction {
                    agent: self.id,
                    dropped,
                });
            }

            let agent = self.id;
            let mut on_delta = |delta: Delta<'_>| match delta {
                Delta::Text(text) => on_event(Event::MessageDelta {
                    agent,
                    text: text.into(),
                }),
                Delta::Reasoning(text) => on_event(Event::ThinkingDelta {
                    agent,
                    text: text.into(),
                }),
                Delta::Code(text) => on_event(Event::CodeDelta {
                    agent,
                    text: text.into(),
                }),
            };
            let left = self.budget.wall_clock.saturating_sub(started.elapsed());
            // Timed here rather than inside the provider, so a step that spent most of its
            // wait backing off between retries says so.
            let asked = Instant::now();
            let completion = tokio::select! {
                biased;
                // Waiting on the model is where a turn spends most of its time, so both
                // the interrupt and the clock have to win a race against it.
                _ = cancelled.changed() => break StopReason::Partial,
                _ = tokio::time::sleep(left) => break StopReason::Partial,
                completion = self.provider.stream(&self.messages, &self.tools, &mut on_delta) => completion?,
            };
            let llm_ms = asked.elapsed().as_millis() as u64;
            let ttft_ms = completion.ttft_ms;
            let reason = completion.stop;
            context = completion.usage.input;
            shape = self.shape(context);

            // Text is the answer only when the model asked for nothing else. Narration
            // alongside a tool call is not an answer, however it reads.
            let answered = completion.tool_calls.is_empty();
            if answered {
                answer = completion.text.clone();
                self.messages
                    .push(Message::text(Role::Assistant, completion.text));
            } else {
                self.messages.push(Message::calls(
                    completion.text,
                    completion.tool_calls.clone(),
                ));
                for call in &completion.tool_calls {
                    let result = self.execute(call, on_event).await?;
                    self.messages
                        .push(Message::tool_result(call.id.clone(), result));
                }
            }

            // Drained after the tools have run, because a tool that talks to a model spends
            // on this step and the agent never sees those calls to bill them itself.
            let mut spent = completion.usage;
            let (batched_in, batched_out) = self.host.drain_tokens();
            spent.input += batched_in;
            spent.output += batched_out;
            usage.input += spent.input;
            usage.output += spent.output;
            if let Some(cached) = spent.cached {
                *usage.cached.get_or_insert(0) += cached;
            }
            on_event(Event::StepEnd {
                turn_id,
                agent: self.id,
                step: steps,
                llm_ms,
                ttft_ms,
                usage: spent,
            });

            if answered {
                break reason;
            }
        };

        if stop == StopReason::Partial && answer.is_empty() {
            answer = format!(
                "Stopped after {steps} steps without reaching an answer. \
                 Ask again to continue from where this left off."
            );
        }
        on_event(Event::Answer {
            agent: self.id,
            text: answer.clone(),
        });
        on_event(Event::TurnEnd {
            turn_id,
            agent: self.id,
            stop,
            usage,
            context,
            wall_ms: started.elapsed().as_millis() as u64,
            shape,
        });
        Ok(Outcome { text: answer, stop })
    }

    /// Splits `context` — the provider's exact prompt total — across the three things that
    /// fill it, by serialized size. There is no tokenizer here and adding one to sharpen a
    /// gauge would be a poor trade: prose and JSON tokenize differently enough that each
    /// bucket is worth about ±10%, which is the precision the question deserves.
    fn shape(&self, context: u32) -> ContextShape {
        let system = self.messages.first().map_or(0, size);
        let tools = serde_json::to_vec(&self.tools).map_or(0, |bytes| bytes.len());
        let said: usize = self.messages.iter().skip(1).map(size).sum();
        let total = system + tools + said;
        if total == 0 || context == 0 {
            return ContextShape::default();
        }
        let share = |part: usize| ((part as u64 * context as u64) / total as u64) as u32;
        ContextShape {
            system: share(system),
            tools: share(tools),
            messages: share(said),
        }
    }

    async fn execute(
        &mut self,
        call: &ToolCall,
        on_event: &mut (dyn FnMut(Event) + Send),
    ) -> Result<String> {
        let args: Value = serde_json::from_str(&call.arguments).unwrap_or(Value::Null);
        on_event(Event::ToolStart {
            agent: self.id,
            call_id: call.id.clone(),
            name: call.name.clone(),
            args: args.clone(),
        });

        // A bad call is data the model can act on, never an exception that kills the turn.
        let refusal = if call.name != "python" {
            Some(format!(
                "ERROR: no tool named `{}`. The only tool is `python`.",
                call.name
            ))
        } else if python_code(&args).is_none() {
            Some("ERROR: arguments must be JSON like {\"code\": \"...\"}.".to_string())
        } else {
            None
        };
        if let Some(message) = refusal {
            on_event(Event::ToolEnd {
                agent: self.id,
                call_id: call.id.clone(),
                ok: false,
                summary: message.clone(),
                ms: 0,
            });
            return Ok(message);
        }

        let agent = self.id;
        let call_id = call.id.clone();
        let started = Instant::now();
        let mut captured = String::new();
        let host = self.host.clone();
        let timeout = self.budget.cell_timeout;
        let code = python_code(&args).unwrap_or_default().to_string();
        // ponytail: a cell that also ignores the interrupt takes the kernel down with it and
        // ends the session. Restarting means rebuilding the namespace, which needs a snapshot.
        let outcome = self
            .kernel()
            .await?
            .exec(
                &code,
                &host,
                &mut |text| {
                    captured.push_str(text);
                    on_event(Event::ToolStream {
                        agent,
                        call_id: call_id.clone(),
                        text: text.into(),
                    });
                },
                timeout,
            )
            .await?;

        let mut result = captured;
        if outcome.ok && !outcome.repr.is_empty() {
            result.push_str(&outcome.repr);
        }
        let result = tool_result(outcome.ok, result, &outcome.traceback);
        if !outcome.names.is_empty() || !outcome.gone.is_empty() {
            on_event(Event::Namespace {
                agent: self.id,
                call_id: call.id.clone(),
                names: outcome.names,
                gone: outcome.gone,
                trimmed: outcome.trimmed,
            });
        }
        on_event(Event::ToolEnd {
            agent: self.id,
            call_id: call.id.clone(),
            ok: outcome.ok,
            summary: match outcome.ok {
                true => summarize(&result),
                false => exception(&outcome.traceback),
            },
            ms: started.elapsed().as_millis() as u64,
        });
        Ok(result)
    }

    /// Oldest tool output goes first: a page can be fetched again, a line of reasoning cannot.
    fn compact(&mut self) -> usize {
        let mut total: usize = self.messages.iter().map(size).sum();
        if total <= self.budget.context_chars {
            return 0;
        }
        let mut dropped = 0;
        for message in &mut self.messages {
            if total <= self.budget.context_chars {
                break;
            }
            if message.role == Role::Tool && message.content.as_deref() != Some(DROPPED) {
                total = total - size(message) + DROPPED.len();
                message.content = Some(DROPPED.to_string());
                dropped += 1;
            }
        }
        dropped
    }
}

/// What a call leaves in the model's context, live or replayed from the log. A failure is
/// data the model can act on, so it is marked rather than raised, and anything past the
/// limit is cut at a character boundary: the byte index alone can land inside one.
fn tool_result(ok: bool, mut result: String, failure: &str) -> String {
    if ok {
        if result.is_empty() {
            result.push_str("(no output)");
        }
    } else {
        result.push_str("ERROR: ");
        result.push_str(failure);
    }
    if result.len() > TOOL_RESULT_LIMIT {
        result.truncate(result.floor_char_boundary(TOOL_RESULT_LIMIT));
        // Which half was cut — printed output or the repr — cannot be told apart once they
        // are one string, so the note claims only what is true of both.
        result.push_str(
            "\n[truncated; the value a cell ends on stays whole in `_`, and anything printed is \
             worth re-deriving by slicing in code]",
        );
    }
    result
}

fn size(message: &Message) -> usize {
    message.content.as_ref().map_or(0, |c| c.len())
        + message
            .tool_calls
            .iter()
            .map(|c| c.arguments.len())
            .sum::<usize>()
}

/// The line a Python reader actually looks for: the exception, not the frames above it.
fn exception(traceback: &str) -> String {
    traceback
        .lines()
        .rev()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("failed")
        .trim()
        .to_string()
}

fn summarize(result: &str) -> String {
    let lines = result.lines().count();
    let first = result.lines().next().unwrap_or_default();
    if lines <= 1 && first.chars().count() <= 80 {
        first.to_string()
    } else {
        format!("{lines} lines · {} chars", result.chars().count())
    }
}

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use corpus_kernel::{Host, Kernel};
use corpus_provider::{Delta, Message, Provider, Role, StopReason, Tool, ToolCall, Usage};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use uuid::Uuid;

const TOOL_RESULT_LIMIT: usize = 8000;
const DROPPED: &str = "[output dropped to save context; run the call again if you still need it]";

/// The session log. The model's context is derived from it, never stored beside it,
/// which is what lets compaction drop tool output without losing the session (§4.1).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "t", rename_all = "snake_case")]
pub enum Event {
    SessionStart {
        session_id: Uuid,
        model: String,
    },
    TurnStart {
        turn_id: Uuid,
        agent: Uuid,
    },
    UserMessage {
        turn_id: Uuid,
        text: String,
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
    Compaction {
        agent: Uuid,
        dropped: usize,
    },
    TurnEnd {
        turn_id: Uuid,
        agent: Uuid,
        stop: StopReason,
        usage: Usage,
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

/// A monotonic counter rather than a flag: a receiver taken at the start of a turn
/// only ever sees interrupts raised after it, so a stale one cannot kill the next turn.
#[derive(Clone)]
struct Cancel(tokio::sync::watch::Sender<u64>);

impl Cancel {
    fn new() -> Cancel {
        Cancel(tokio::sync::watch::channel(0).0)
    }

    fn raise(&self) {
        self.0.send_modify(|count| *count += 1);
    }

    fn watch(&self) -> tokio::sync::watch::Receiver<u64> {
        self.0.subscribe()
    }
}

pub struct Outcome {
    pub text: String,
    pub stop: StopReason,
    pub usage: Usage,
}

pub struct Agent {
    pub id: Uuid,
    cancel: Cancel,
    provider: Provider,
    kernel: Kernel,
    host: Arc<dyn Host>,
    messages: Vec<Message>,
    tools: Vec<Tool>,
    budget: Budget,
}

impl Agent {
    pub fn new(
        provider: Provider,
        kernel: Kernel,
        host: Arc<dyn Host>,
        system_prompt: &str,
        budget: Budget,
    ) -> Agent {
        Agent {
            id: Uuid::now_v7(),
            cancel: Cancel::new(),
            provider,
            kernel,
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
        let kernel = self.kernel.interrupter();
        Interrupt::new(move || {
            cancel.raise();
            kernel.raise();
        })
    }

    /// Rebuilds the model's context from the session log. The kernel namespace does not
    /// come back: variables from the earlier run are gone and the model must refetch.
    pub fn replay(&mut self, events: impl IntoIterator<Item = Event>) {
        let mut pending: Option<ToolCall> = None;
        let mut output = String::new();
        for event in events {
            match event {
                Event::UserMessage { text, .. } => {
                    self.messages.push(Message::text(Role::User, text))
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
        let turn_id = Uuid::now_v7();
        on_event(Event::TurnStart {
            turn_id,
            agent: self.id,
        });
        on_event(Event::UserMessage {
            turn_id,
            text: prompt.to_string(),
        });
        self.messages.push(Message::text(Role::User, prompt));

        let started = Instant::now();
        let mut cancelled = self.cancel.watch();
        let mut usage = Usage::default();
        let mut context = 0;
        let mut answer = String::new();
        let mut steps = 0;

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
            let completion = tokio::select! {
                biased;
                // Waiting on the model is where a turn spends most of its time, so both
                // the interrupt and the clock have to win a race against it.
                _ = cancelled.changed() => break StopReason::Partial,
                _ = tokio::time::sleep(left) => break StopReason::Partial,
                completion = self.provider.stream(&self.messages, &self.tools, &mut on_delta) => completion?,
            };
            usage.input += completion.usage.input;
            usage.output += completion.usage.output;
            context = completion.usage.input;

            if completion.tool_calls.is_empty() {
                // Text is the answer only when the model asked for nothing else (§6).
                // Narration alongside a tool call is not an answer, however it reads.
                answer = completion.text.clone();
                self.messages
                    .push(Message::text(Role::Assistant, completion.text));
                break completion.stop;
            }

            self.messages.push(Message::calls(
                completion.text,
                completion.tool_calls.clone(),
            ));
            for call in &completion.tool_calls {
                let result = self.execute(call, on_event).await?;
                self.messages
                    .push(Message::tool_result(call.id.clone(), result));
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
        });
        Ok(Outcome {
            text: answer,
            stop,
            usage,
        })
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

        // A bad call is data the model can act on, never an exception that kills the turn (§9.8).
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
        // ponytail: a cell that also ignores the interrupt takes the kernel down with it and
        // ends the session. Restarting means rebuilding the namespace, which needs a snapshot.
        let outcome = self
            .kernel
            .exec(
                python_code(&args).unwrap_or_default(),
                self.host.as_ref(),
                &mut |text| {
                    captured.push_str(text);
                    on_event(Event::ToolStream {
                        agent,
                        call_id: call_id.clone(),
                        text: text.into(),
                    });
                },
                self.budget.cell_timeout,
            )
            .await?;

        let mut result = captured;
        if outcome.ok && !outcome.repr.is_empty() {
            result.push_str(&outcome.repr);
        }
        let result = tool_result(outcome.ok, result, &outcome.traceback);
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

    /// Oldest tool output goes first: a page can be fetched again, a line of reasoning cannot (§6).
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
        result.push_str("\n[truncated; keep the rest in a variable and process it in code]");
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

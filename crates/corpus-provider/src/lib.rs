use std::collections::BTreeMap;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize, Serializer};
use serde_json::{Map, Value, json};

// Control tokens are stripped from the stream rather than sent as `stop` strings: this
// gateway's engine mangles a streaming answer when `stop` is set, cutting the text
// mid-word and spilling a character into the wrong field. Stopping generation at those
// tokens is the server's job, and doing it here costs more than it buys.

/// Reasoning models wrap their thinking in this. A provider that splits it into the
/// `reasoning` field imperfectly leaves the tail, and the terminator, sitting in the
/// answer — so whatever arrives before it is thinking, not answer.
const THOUGHT_END: &str = "</think>";

/// How much answer to hold while a `</think>` might still arrive. Long enough for the
/// tail of a thought, short enough that a real answer is not visibly delayed.
const HOLD: usize = 200;

/// Silence, not slowness. While bytes keep arriving the clock keeps resetting, so a
/// long answer is safe and a dead connection is not (§9.3). The window that has to fit
/// under this is the one before the first byte: a reasoning model prefills and thinks
/// with nothing on the wire, and at ~100k of context that is minutes, not seconds.
const SILENCE: Duration = Duration::from_secs(300);
const CONNECT: Duration = Duration::from_secs(15);

const ATTEMPTS: u32 = 3;
const BACKOFF: Duration = Duration::from_millis(200);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    Stop,
    ToolCalls,
    Length,
    /// A budget ran out; the answer is the best partial one we had.
    Partial,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub input: u32,
    pub output: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Message {
    pub role: Role,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    pub fn text(role: Role, content: impl Into<String>) -> Message {
        Message {
            role,
            content: Some(content.into()),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }

    pub fn calls(text: String, tool_calls: Vec<ToolCall>) -> Message {
        Message {
            role: Role::Assistant,
            content: (!text.is_empty()).then_some(text),
            tool_calls,
            tool_call_id: None,
        }
    }

    pub fn tool_result(call_id: impl Into<String>, content: impl Into<String>) -> Message {
        Message {
            role: Role::Tool,
            content: Some(content.into()),
            tool_calls: Vec::new(),
            tool_call_id: Some(call_id.into()),
        }
    }
}

/// `id` is minted by the provider and must travel back untouched (§4.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

impl Serialize for ToolCall {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        json!({
            "id": self.id,
            "type": "function",
            "function": { "name": self.name, "arguments": self.arguments },
        })
        .serialize(s)
    }
}

#[derive(Debug, Clone)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

impl Serialize for Tool {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        })
        .serialize(s)
    }
}

#[derive(Debug)]
pub enum Delta<'a> {
    Text(&'a str),
    Reasoning(&'a str),
    /// The cell being written, decoded out of the call's arguments as they arrive. A long
    /// one takes minutes to generate, and a screen that shows nothing until it lands reads
    /// as a hang rather than as work.
    Code(&'a str),
}

#[derive(Debug)]
pub struct Completion {
    pub text: String,
    pub reasoning: String,
    pub tool_calls: Vec<ToolCall>,
    pub stop: StopReason,
    pub usage: Usage,
}

pub struct Provider {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
    pub model: String,
}

impl Provider {
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Provider {
        Provider {
            http: reqwest::Client::builder()
                .read_timeout(SILENCE)
                .connect_timeout(CONNECT)
                .build()
                .expect("http client"),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key: api_key.into(),
            model: model.into(),
        }
    }

    /// What this provider offers, in the order it lists them.
    pub async fn models(&self) -> Result<Vec<String>> {
        let response = self
            .http
            .get(format!("{}/models", self.base_url))
            .bearer_auth(&self.api_key)
            .send()
            .await
            .context("could not reach the provider")?;
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        if !status.is_success() {
            bail!("provider returned {status}: {}", body.trim());
        }
        let listing: Value = serde_json::from_str(&body)
            .with_context(|| format!("provider sent an unreadable model list: {}", body.trim()))?;
        Ok(listing["data"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|model| model["id"].as_str().map(String::from))
            .collect())
    }

    pub async fn stream(
        &self,
        messages: &[Message],
        tools: &[Tool],
        on_delta: &mut (dyn FnMut(Delta<'_>) + Send),
    ) -> Result<Completion> {
        let mut body = Map::new();
        body.insert("model".into(), json!(self.model));
        body.insert("messages".into(), json!(messages));
        body.insert("stream".into(), json!(true));
        body.insert("stream_options".into(), json!({ "include_usage": true }));
        if !tools.is_empty() {
            body.insert("tools".into(), json!(tools));
        }
        let body = Value::Object(body);

        for attempt in 0..ATTEMPTS {
            match self.attempt(&body, on_delta).await {
                Ok(completion) => return Ok(completion),
                Err(failure) if failure.retryable && attempt + 1 < ATTEMPTS => {
                    tokio::time::sleep(BACKOFF * 2u32.pow(attempt)).await;
                }
                Err(failure) => return Err(failure.error),
            }
        }
        unreachable!("the last attempt returns rather than retrying")
    }

    async fn attempt(
        &self,
        body: &Value,
        on_delta: &mut (dyn FnMut(Delta<'_>) + Send),
    ) -> Result<Completion, Failure> {
        let response = self
            .http
            .post(format!("{}/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .json(body)
            .send()
            .await
            .map_err(Failure::transport)?;

        let status = response.status();
        if !status.is_success() {
            let detail = response.text().await.unwrap_or_default();
            return Err(Failure {
                retryable: status.as_u16() == 429 || status.is_server_error(),
                error: anyhow::anyhow!("provider returned {status}: {}", detail.trim()),
            });
        }

        let mut acc = Acc::default();
        let mut line = Vec::new();
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(Failure::transport)?;
            // A line at a time, not a byte at a time: an event arrives split across chunks
            // as often as not, so what crosses a boundary is kept and completed.
            let mut rest: &[u8] = chunk.as_ref();
            while let Some(at) = rest.iter().position(|byte| *byte == b'\n') {
                line.extend_from_slice(&rest[..at]);
                rest = &rest[at + 1..];
                if acc.feed(&line, on_delta) {
                    acc.flush(on_delta);
                    return Ok(acc.finish());
                }
                line.clear();
            }
            line.extend_from_slice(rest);
        }
        acc.feed(&line, on_delta);
        acc.flush(on_delta);
        Ok(acc.finish())
    }
}

struct Failure {
    retryable: bool,
    error: anyhow::Error,
}

impl Failure {
    fn transport(error: reqwest::Error) -> Failure {
        Failure {
            retryable: true,
            error: anyhow::Error::new(error).context("provider stream broke"),
        }
    }
}

/// Strips control tokens from a stream, holding back a tail that could be the start of
/// one split across two deltas.
#[derive(Default)]
struct Clean {
    held: String,
}

impl Clean {
    fn feed(&mut self, chunk: &str) -> String {
        self.held.push_str(chunk);
        let mut out = String::new();
        loop {
            let Some(start) = self.held.find("<|") else {
                let keep = usize::from(self.held.ends_with('<'));
                let split = self.held.len() - keep;
                out.push_str(&self.held[..split]);
                self.held.drain(..split);
                return out;
            };
            out.push_str(&self.held[..start]);
            self.held.drain(..start);
            let Some(end) = self.held.find("|>") else {
                return out;
            };
            self.held.drain(..end + 2);
        }
    }

    /// Whatever is left at the end of the stream, unless it is a token that never closed.
    fn flush(&mut self) -> String {
        let held = std::mem::take(&mut self.held);
        if held.starts_with("<|") {
            String::new()
        } else {
            held
        }
    }
}

/// Reads the `code` argument out of a call while its JSON is still arriving. What it
/// returns is for the screen only: the arguments the agent acts on are the accumulated
/// string, parsed once the call is whole, so a fragment this cannot decode costs a
/// glimpse of code rather than the cell itself.
#[derive(Default)]
struct Written {
    held: String,
    open: bool,
    closed: bool,
}

impl Written {
    /// The text this fragment added, empty until the value itself starts.
    fn feed(&mut self, fragment: &str) -> String {
        if self.closed {
            return String::new();
        }
        self.held.push_str(fragment);
        if !self.open && !self.start() {
            return String::new();
        }
        let mut out = String::new();
        let mut rest = self.held.as_str();
        loop {
            let mut chars = rest.chars();
            match chars.next() {
                None => break,
                Some('"') => {
                    self.closed = true;
                    rest = chars.as_str();
                    break;
                }
                // An escape split across two fragments is left whole for the next one:
                // half of one decodes to nothing anybody wants to read.
                Some('\\') => match unescape(chars.as_str()) {
                    Some((ch, tail)) => {
                        out.push(ch);
                        rest = tail;
                    }
                    None => break,
                },
                Some(ch) => {
                    out.push(ch);
                    rest = chars.as_str();
                }
            }
        }
        let consumed = self.held.len() - rest.len();
        self.held.drain(..consumed);
        out
    }

    /// Finds the opening quote of the `code` value and drops everything up to it.
    fn start(&mut self) -> bool {
        const KEY: &str = "\"code\"";
        let Some(key) = self.held.find(KEY) else {
            return false;
        };
        let after = key + KEY.len();
        let Some(quote) = self.held[after..].find('"') else {
            return false;
        };
        self.held.drain(..after + quote + 1);
        self.open = true;
        true
    }
}

/// One JSON escape, without its backslash, and whatever follows it. `None` while the
/// escape is still incomplete, so the caller can wait for the rest of it.
fn unescape(text: &str) -> Option<(char, &str)> {
    let mut chars = text.chars();
    let ch = chars.next()?;
    if ch != 'u' {
        let decoded = match ch {
            'n' => '\n',
            't' => '\t',
            'r' => '\r',
            'b' => '\u{8}',
            'f' => '\u{c}',
            // `"`, `\` and `/` stand for themselves.
            other => other,
        };
        return Some((decoded, chars.as_str()));
    }
    let rest = chars.as_str();
    let point = u32::from_str_radix(rest.get(..4)?, 16).ok()?;
    // ponytail: half of a surrogate pair shows as one replacement character, so an emoji
    // written this way is a smudge on screen until the halves are joined here.
    let ch = char::from_u32(point).unwrap_or(char::REPLACEMENT_CHARACTER);
    Some((ch, &rest[4..]))
}

#[derive(Default)]
struct PartialCall {
    id: String,
    name: String,
    arguments: String,
    written: Written,
}

#[derive(Default)]
struct Acc {
    text: String,
    reasoning: String,
    clean_text: Clean,
    clean_reasoning: Clean,
    withheld: String,
    settled: bool,
    calls: BTreeMap<u64, PartialCall>,
    stop: Option<StopReason>,
    usage: Usage,
}

impl Acc {
    /// Returns true when the stream announced its end.
    fn feed(&mut self, line: &[u8], on_delta: &mut dyn FnMut(Delta<'_>)) -> bool {
        let Ok(line) = std::str::from_utf8(line) else {
            return false;
        };
        let Some(payload) = line.trim().strip_prefix("data:") else {
            return false;
        };
        let payload = payload.trim();
        if payload == "[DONE]" {
            return true;
        }
        let Ok(chunk) = serde_json::from_str::<Value>(payload) else {
            return false;
        };

        if let Some(usage) = chunk.get("usage").filter(|u| u.is_object()) {
            self.usage.input = usage["prompt_tokens"].as_u64().unwrap_or(0) as u32;
            self.usage.output = usage["completion_tokens"].as_u64().unwrap_or(0) as u32;
        }
        for choice in chunk["choices"].as_array().into_iter().flatten() {
            if let Some(reason) = choice["finish_reason"].as_str() {
                self.stop = Some(match reason {
                    "tool_calls" | "function_call" => StopReason::ToolCalls,
                    "length" => StopReason::Length,
                    _ => StopReason::Stop,
                });
            }
            let delta = &choice["delta"];
            if let Some(text) = delta["content"].as_str().filter(|t| !t.is_empty()) {
                let text = self.clean_text.feed(text);
                if !text.is_empty() {
                    self.answer(text, on_delta);
                }
            }
            // Some models put the whole answer in `reasoning`; reading only `content`
            // turns those turns into silence (§9.2).
            for key in ["reasoning", "reasoning_content"] {
                if let Some(text) = delta[key].as_str().filter(|t| !t.is_empty()) {
                    let text = self.clean_reasoning.feed(text);
                    if !text.is_empty() {
                        self.reasoning.push_str(&text);
                        on_delta(Delta::Reasoning(&text));
                    }
                }
            }
            for call in delta["tool_calls"].as_array().into_iter().flatten() {
                // Calls arrive split across deltas and are keyed by index, not by order:
                // the name lands in the first delta, the arguments in the ones after (§9.4).
                let slot = self
                    .calls
                    .entry(call["index"].as_u64().unwrap_or(0))
                    .or_default();
                for (field, into) in [("id", &mut slot.id), ("name", &mut slot.name)] {
                    let value = call[field]
                        .as_str()
                        .or_else(|| call["function"][field].as_str());
                    if into.is_empty()
                        && let Some(value) = value
                    {
                        *into = value.to_string();
                    }
                }
                if let Some(args) = call["function"]["arguments"].as_str() {
                    slot.arguments.push_str(args);
                    // ponytail: one cell at a time on screen. Two calls streamed together
                    // interleave their code here; keeping them apart means carrying the
                    // index on the delta and a block per call in whatever draws it.
                    let written = slot.written.feed(args);
                    if !written.is_empty() {
                        on_delta(Delta::Code(&written));
                    }
                }
            }
        }
        false
    }

    /// Everything up to a `</think>` was thinking, however it arrived; what follows the
    /// terminator is where the answer starts.
    fn cut_thought(&mut self, buffer: &str) -> Option<String> {
        let at = buffer.find(THOUGHT_END)?;
        self.reasoning
            .push_str(&buffer[..at].replace("<think>", ""));
        Some(buffer[at + THOUGHT_END.len()..].to_string())
    }

    fn answer(&mut self, text: String, on_delta: &mut dyn FnMut(Delta<'_>)) {
        // Only a response that has already streamed reasoning can still have its answer
        // turn out to be the tail of a thought.
        if self.settled || self.reasoning.is_empty() {
            self.text.push_str(&text);
            on_delta(Delta::Text(&text));
            return;
        }
        self.withheld.push_str(&text);
        let held = std::mem::take(&mut self.withheld);
        let ready = match self.cut_thought(&held) {
            Some(tail) => tail,
            None if held.len() > HOLD => held,
            None => {
                self.withheld = held;
                return;
            }
        };
        self.settled = true;
        if !ready.is_empty() {
            self.text.push_str(&ready);
            on_delta(Delta::Text(&ready));
        }
    }

    /// The stream ended and no terminator ever came, so what was held back has to be
    /// placed somewhere. A model that asked for a tool was not answering, so anything
    /// held back there belongs to the thinking; otherwise it is the answer. Either way
    /// it is shown, because text in the log that never reached the screen is worse than
    /// no text.
    fn flush(&mut self, on_delta: &mut dyn FnMut(Delta<'_>)) {
        let held = std::mem::take(&mut self.withheld);
        if held.is_empty() {
            return;
        }
        self.settled = true;
        if self.calls.is_empty() {
            self.text.push_str(&held);
            on_delta(Delta::Text(&held));
        } else {
            self.reasoning.push_str(&held);
            on_delta(Delta::Reasoning(&held));
        }
    }

    /// Always after `flush`, which is what places whatever was held back.
    fn finish(mut self) -> Completion {
        self.text.push_str(&self.clean_text.flush());
        // A terminator that arrived after the hold gave up still marks where thinking ended.
        let text = std::mem::take(&mut self.text);
        self.text = self.cut_thought(&text).unwrap_or(text);
        self.reasoning.push_str(&self.clean_reasoning.flush());
        let tool_calls: Vec<ToolCall> = self
            .calls
            .into_values()
            .map(|c| ToolCall {
                id: c.id,
                name: c.name,
                arguments: c.arguments,
            })
            .collect();
        let mut text = self.text;
        if text.is_empty() && tool_calls.is_empty() {
            text = self.reasoning.clone();
        }
        let stop = self.stop.unwrap_or(if tool_calls.is_empty() {
            StopReason::Stop
        } else {
            StopReason::ToolCalls
        });
        Completion {
            text,
            reasoning: self.reasoning,
            tool_calls,
            stop,
            usage: self.usage,
        }
    }
}

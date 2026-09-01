use std::collections::BTreeMap;
use std::io::Write;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize, Serializer};
use serde_json::{Value, json};

pub mod anthropic;
mod route;

pub use route::{DEFAULT_MAX_TOKENS, FALLBACK, Protocol, Registry, Route, RouteModel};

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
/// long answer is safe and a dead connection is not. The window that has to fit
/// under this is the one before the first byte: a reasoning model prefills and thinks
/// with nothing on the wire, and at ~100k of context that is minutes, not seconds.
const SILENCE: Duration = Duration::from_secs(300);
const CONNECT: Duration = Duration::from_secs(15);

/// The version this crate's translation was written against. Anthropic requires the
/// header and treats its absence as an error rather than as "the latest".
const ANTHROPIC_VERSION: &str = "2023-06-01";

const ATTEMPTS: u32 = 3;
const BACKOFF: Duration = Duration::from_millis(200);

/// RFC 3339 down to the millisecond, which is the grain these questions are asked at.
///
/// ponytail: written in UTC, because `now_local` is the call that fails in a process with
/// threads, and this one has plenty. Whoever reads the line converts.
fn now() -> String {
    let at = time::OffsetDateTime::now_utc();
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
        at.year(),
        u8::from(at.month()),
        at.day(),
        at.hour(),
        at.minute(),
        at.second(),
        at.millisecond(),
    )
}

/// A log line that opens with the moment it was written. Spliced onto the front rather
/// than built through a map, which would sort the keys and bury the interesting one.
fn stamped<T: Serialize>(value: &T) -> String {
    let body = serde_json::to_string(value).unwrap_or_default();
    match body.strip_prefix('{').filter(|rest| *rest != "}") {
        Some(rest) => format!("{{\"at\":\"{}\",{rest}", now()),
        None => format!("{{\"at\":\"{}\"}}", now()),
    }
}

/// An append-only file of JSON lines, each stamped with the moment it was written. The
/// wire trace and the session log are both exactly this, so they open and write it the
/// same way rather than each spelling the same four steps out.
pub struct Jsonl(std::sync::Mutex<std::fs::File>);

impl Jsonl {
    pub fn to(path: &std::path::Path) -> Result<Jsonl> {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("cannot write {}", path.display()))?;
        Ok(Jsonl(std::sync::Mutex::new(file)))
    }

    /// Reads back what [`Jsonl::to`] wrote. A line that will not parse is skipped rather
    /// than failing the read: a log is worth what can still be recovered from it, and the
    /// last line of one whose writer was killed mid-write is routinely half a line.
    pub fn read<T: serde::de::DeserializeOwned>(path: &std::path::Path) -> Result<Vec<T>> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("cannot read {}", path.display()))?;
        Ok(text
            .lines()
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect())
    }

    /// Never fails whatever it is recording: a line that cannot be written is worth less
    /// than the turn it would have interrupted. One `write_all` rather than `writeln!`,
    /// which splits its format into a syscall per piece: this is on the path of every
    /// streamed token.
    pub fn record<T: Serialize>(&self, entry: &T) {
        let mut line = stamped(entry);
        line.push('\n');
        if let Ok(mut file) = self.0.lock() {
            let _ = file.write_all(line.as_bytes());
        }
    }
}

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
    /// Prompt tokens the provider served from its own cache. `None` is a provider that
    /// never said, which is not the same as a cache that missed: rendering it as zero
    /// would assert a cold cache nobody measured. Every gateway spells it differently and
    /// the reference one spells it not at all.
    #[serde(default)]
    pub cached: Option<u32>,
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

/// `id` is minted by the provider and must travel back untouched.
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

/// One completion request, borrowing the conversation it asks about.
#[derive(Serialize)]
struct Request<'a> {
    model: &'a str,
    messages: &'a [Message],
    stream: bool,
    stream_options: StreamOptions,
    #[serde(skip_serializing_if = "<[Tool]>::is_empty")]
    tools: &'a [Tool],
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<&'a str>,
}

#[derive(Serialize, Clone, Copy)]
struct StreamOptions {
    include_usage: bool,
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
    /// Request sent to the first delta a reader would have seen, on the attempt that
    /// answered. Retries are the caller's to account for: it is the one that knows it
    /// waited. `None` when the stream ended without ever saying anything.
    pub ttft_ms: Option<u64>,
}

/// The names an OpenAI-compatible listing gives the context window. Every gateway spells
/// it differently and some nest it under `limit`; only OpenAI itself states nothing at
/// all, and a model that names none reports zero.
const WINDOW_KEYS: [&str; 5] = [
    "context_window",
    "contextWindow",
    "context_length",
    "max_model_len",
    "max_context_length",
];

/// Windows for weights whose gateway publishes none, under the names the weights are
/// released as. Deliberately short: a wrong number reads as certainty and throws the
/// readout off, so anything not matched here shows no share at all. The nemotron figure
/// is what a gateway serving it enforces as `max_model_len`, which is also what NVIDIA
/// states for BF16; the same weights reach 1M under NVFP4 on Blackwell.
const KNOWN_WINDOWS: [(&str, u32); 2] = [
    ("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16", 262_144),
    ("x-ai/grok-2", 131_072),
];

/// The window a gateway's own name for a model implies. Gateways rename what they serve,
/// so `nemotron-3-ultra` has to find the weights it is a nickname for: a name matches
/// when its words appear in order inside a released name. Several may match, because a
/// short nickname can sit inside a whole family — the window is only borrowed when every
/// match agrees on it, since a share against the wrong model is worse than no share.
fn known_window(id: &str) -> u32 {
    let words = |name: &str| -> Vec<String> {
        name.split(|letter: char| !letter.is_ascii_alphanumeric())
            .filter(|word| !word.is_empty())
            .map(str::to_ascii_lowercase)
            .collect()
    };
    let alias = words(id);
    // One word names a family rather than a checkpoint, and families do not share a
    // window: `grok` would otherwise borrow whatever grok-2 happens to take.
    if alias.len() < 2 {
        return 0;
    }
    let mut matched = KNOWN_WINDOWS
        .iter()
        .filter(|(known, _)| words(known).windows(alias.len()).any(|run| run == alias))
        .map(|(_, window)| *window);
    let first = matched.next().unwrap_or(0);
    match matched.all(|window| window == first) {
        true => first,
        false => 0,
    }
}

/// One entry of the provider's model listing.
#[derive(Debug, Clone)]
pub struct Model {
    pub id: String,
    /// Tokens the model takes at once, or 0 when the provider does not say.
    pub window: u32,
}

#[derive(Clone)]
pub struct Provider {
    /// Cloned rather than built a second time wherever another agent needs one: the clone
    /// shares this one's connection pool and its trace file, which is what a child of this
    /// session should be speaking through anyway.
    http: reqwest::Client,
    base_url: String,
    api_key: String,
    pub model: String,
    /// Which dialect this endpoint speaks. Everything above this crate writes OpenAI
    /// completions whatever the answer is; the translation lives in `anthropic`.
    protocol: Protocol,
    /// The ceiling a translated request carries. OpenAI treats an absent one as the
    /// model's own maximum and Anthropic refuses the request outright, so a number has to
    /// exist for a route that may be either.
    max_tokens: u32,
    /// Tokens the model may spend thinking, when a reasoning level was selected and this
    /// route spells that level as a budget.
    thinking_budget: Option<u32>,
    /// The wire spelling of a selected reasoning level on an OpenAI-dialect route.
    reasoning_effort: Option<String>,
    headers: BTreeMap<String, String>,
    /// What the inference server actually sent, before anything here has read it. The
    /// session log holds this crate's interpretation of the stream — control tokens
    /// stripped, thinking split from answer, text held back and re-filed — which is the
    /// wrong evidence to take to whoever runs the server. This is the wire itself, and it
    /// is the file to attach to a bug report. Shared, so every provider built against one
    /// endpoint writes to the one file: a batch's traffic is the traffic a batch bug
    /// report needs.
    trace: Option<Arc<Jsonl>>,
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
            protocol: Protocol::OpenAiCompletions,
            max_tokens: DEFAULT_MAX_TOKENS,
            thinking_budget: None,
            reasoning_effort: None,
            headers: BTreeMap::new(),
            trace: None,
        }
    }

    /// A provider built against one route of the table, for one model it serves.
    pub fn on(route: &Route, model: &str) -> Provider {
        let mut provider = Provider::new(&route.base_url, &route.api_key, model);
        provider.protocol = route.protocol;
        provider.max_tokens = route.max_tokens_for(model);
        provider.headers = route.headers.clone();
        provider
    }

    /// Ask this model to think at the named level, in whatever spelling its route uses.
    /// An unoffered level fails here, before the request is built: a turn that silently
    /// did not think costs the same as one that did and answers worse.
    pub fn reasoning(mut self, route: &Route, level: &str) -> Result<Provider> {
        let spelling = match route.effort_wire(&self.model, level) {
            Some(spelling) => spelling,
            // A route that spelled out no entry for this model has said nothing about
            // which levels it takes, so the caller's word is the only one there is and it
            // travels as written. That is the ordinary case behind a router — the run's
            // shim holds the catalogue, not this process — and refusing here would make
            // the level unusable exactly where it is configured.
            None if route.model(&self.model).is_none() => Some(level),
            None => {
                let offered = route
                    .model(&self.model)
                    .map(|entry| entry.reasoning_efforts.keys().cloned().collect::<Vec<_>>())
                    .unwrap_or_default();
                bail!(
                    "model {:?} on route {:?} does not offer the reasoning level \
                     {level:?}; it offers {}",
                    self.model,
                    route.provider,
                    match offered.is_empty() {
                        true => "none".to_string(),
                        false => offered.join(", "),
                    }
                );
            }
        };
        match self.protocol {
            // The only dial the Messages API has for thinking is how many tokens to
            // spend on it, so that is what a level spells out on such a route.
            Protocol::AnthropicMessages => {
                self.thinking_budget = spelling.and_then(|value| value.parse().ok())
            }
            Protocol::OpenAiCompletions => {
                self.reasoning_effort = spelling.map(str::to_string)
            }
        }
        Ok(self)
    }

    /// Records the wire for as long as the provider lives.
    pub fn tracing_to(mut self, trace: Arc<Jsonl>) -> Provider {
        self.trace = Some(trace);
        self
    }

    /// What this provider offers, in the order it lists them.
    pub async fn models(&self) -> Result<Vec<Model>> {
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
            .filter_map(|model| {
                let id = model["id"].as_str()?.to_string();
                let stated = WINDOW_KEYS
                    .iter()
                    .find_map(|key| model[key].as_u64())
                    .or_else(|| model["limit"]["context"].as_u64());
                let window = match stated {
                    Some(tokens) => tokens as u32,
                    None => known_window(&id),
                };
                Some(Model { id, window })
            })
            .collect())
    }

    pub async fn stream(
        &self,
        messages: &[Message],
        tools: &[Tool],
        on_delta: &mut (dyn FnMut(Delta<'_>) + Send),
    ) -> Result<Completion> {
        // Written straight to the bytes that get posted, borrowing the conversation rather
        // than copying it into a `Value` first. This runs once per step of every turn with
        // the whole context in it, so a spare copy of it is a spare copy per step.
        let body = match self.protocol {
            Protocol::OpenAiCompletions => serde_json::to_vec(&Request {
                model: &self.model,
                messages,
                stream: true,
                stream_options: StreamOptions {
                    include_usage: true,
                },
                tools,
                reasoning_effort: self.reasoning_effort.as_deref(),
            }),
            Protocol::AnthropicMessages => serde_json::to_vec(&anthropic::request(
                &self.model,
                messages,
                tools,
                self.max_tokens,
                self.thinking_budget,
                true,
            )),
        }
        .context("serializing the request")?;

        for attempt in 0..ATTEMPTS {
            match self
                .attempt(&body, messages.len(), tools.len(), on_delta)
                .await
            {
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
        body: &[u8],
        messages: usize,
        tools: usize,
        on_delta: &mut (dyn FnMut(Delta<'_>) + Send),
    ) -> Result<Completion, Failure> {
        let url = match self.protocol {
            Protocol::OpenAiCompletions => format!("{}/chat/completions", self.base_url),
            Protocol::AnthropicMessages => format!("{}/messages", self.base_url),
        };
        if let Some(trace) = &self.trace {
            // The body itself stays out: it carries the whole conversation. What a report
            // needs from this side is which model was asked, and how big the ask was.
            trace.record(&json!({ "post": {
                "url": url,
                "model": self.model,
                "bytes": body.len(),
                "messages": messages,
                "tools": tools,
            }}));
        }
        let sent = Instant::now();
        let mut request = self
            .http
            .post(&url)
            .header(reqwest::header::CONTENT_TYPE, "application/json");
        request = match self.protocol {
            Protocol::OpenAiCompletions => request.bearer_auth(&self.api_key),
            // Anthropic authenticates with its own header and treats a missing version as
            // an error rather than as "the latest one".
            Protocol::AnthropicMessages => request
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", ANTHROPIC_VERSION),
        };
        for (name, value) in &self.headers {
            request = request.header(name, value);
        }
        let mut response = request
            .body(body.to_vec())
            .send()
            .await
            .map_err(Failure::transport)?;

        let status = response.status();
        if let Some(trace) = &self.trace {
            // Stamped where the headers land, so the wait before it is the model's own.
            trace.record(&json!({ "status": status.as_u16() }));
        }
        if !status.is_success() {
            let detail = response.text().await.unwrap_or_default();
            return Err(Failure {
                retryable: status.as_u16() == 429 || status.is_server_error(),
                error: anyhow::anyhow!("provider returned {status}: {}", detail.trim()),
            });
        }

        let mut acc = Acc::default();
        let mut dialect = Dialect::of(self.protocol);
        let mut line = Vec::new();
        let mut read = 0;
        // Which read a line came off is what tells a buffered server from a streaming one:
        // equal timestamps could be luck, one read cannot.
        let trace = self.trace.as_deref();
        let traced = |read: u32, line: &[u8]| {
            if let Some(trace) = trace
                && !line.is_empty()
            {
                trace.record(&json!({ "read": read, "line": String::from_utf8_lossy(line) }));
            }
        };
        while let Some(chunk) = response.chunk().await.map_err(Failure::transport)? {
            read += 1;
            // A line at a time, not a byte at a time: an event arrives split across chunks
            // as often as not, so what crosses a boundary is kept and completed.
            let mut rest: &[u8] = chunk.as_ref();
            while let Some(at) = rest.iter().position(|byte| *byte == b'\n') {
                line.extend_from_slice(&rest[..at]);
                rest = &rest[at + 1..];
                traced(read, &line);
                if dialect.feed(&line, &mut acc, on_delta) {
                    acc.flush(on_delta);
                    return Ok(acc.finish(sent));
                }
                line.clear();
            }
            line.extend_from_slice(rest);
        }
        // Whatever the stream ended on without a newline of its own.
        traced(read, &line);
        dialect.feed(&line, &mut acc, on_delta);
        acc.flush(on_delta);
        Ok(acc.finish(sent))
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

/// Which protocol's stream is being read. The accumulator only ever sees OpenAI chunks,
/// so this is the whole of what a second protocol costs on the reading side.
enum Dialect {
    OpenAi,
    Anthropic(anthropic::Stream),
}

impl Dialect {
    fn of(protocol: Protocol) -> Dialect {
        match protocol {
            Protocol::OpenAiCompletions => Dialect::OpenAi,
            Protocol::AnthropicMessages => Dialect::Anthropic(anthropic::Stream::default()),
        }
    }

    /// Returns true when the stream announced its end.
    fn feed(
        &mut self,
        line: &[u8],
        acc: &mut Acc,
        on_delta: &mut dyn FnMut(Delta<'_>),
    ) -> bool {
        let stream = match self {
            Dialect::OpenAi => return acc.feed(line, on_delta),
            Dialect::Anthropic(stream) => stream,
        };
        let Ok(line) = std::str::from_utf8(line) else {
            return false;
        };
        // `event:` lines are ignored: every Anthropic frame repeats its type inside the
        // JSON, so the data line alone carries everything.
        let Some(payload) = line.trim().strip_prefix("data:") else {
            return false;
        };
        let Ok(event) = serde_json::from_str::<Value>(payload.trim()) else {
            return false;
        };
        let (chunks, done) = stream.feed(&event);
        for chunk in &chunks {
            acc.chunk(chunk, on_delta);
        }
        done
    }
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
    /// When the model first said something a reader could see. Latched where a delta is
    /// actually emitted, not where one looks likely: a frame carrying only a control
    /// token, or only the name of a call whose arguments have not started, reaches nobody
    /// and must not start the clock.
    first_token: Option<Instant>,
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
        self.chunk(&chunk, on_delta);
        false
    }

    /// One OpenAI-shaped chunk, whoever shaped it: the provider that sent it, or the
    /// translation in [`anthropic`] for a route that speaks the other protocol.
    fn chunk(&mut self, chunk: &Value, on_delta: &mut dyn FnMut(Delta<'_>)) {
        let mut spoke = false;

        if let Some(usage) = chunk.get("usage").filter(|u| u.is_object()) {
            self.usage.input = usage["prompt_tokens"].as_u64().unwrap_or(0) as u32;
            self.usage.output = usage["completion_tokens"].as_u64().unwrap_or(0) as u32;
            // Three spellings for one number, in the order the gateways we speak to use
            // them. Absent everywhere leaves it None.
            self.usage.cached = usage["prompt_tokens_details"]["cached_tokens"]
                .as_u64()
                .or_else(|| usage["prompt_cache_hit_tokens"].as_u64())
                .or_else(|| usage["cache_read_input_tokens"].as_u64())
                .map(|cached| cached as u32);
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
                    spoke |= self.answer(text, on_delta);
                }
            }
            // Some models put the whole answer in `reasoning`; reading only `content`
            // turns those turns into silence.
            for key in ["reasoning", "reasoning_content"] {
                if let Some(text) = delta[key].as_str().filter(|t| !t.is_empty()) {
                    let text = self.clean_reasoning.feed(text);
                    if !text.is_empty() {
                        self.reasoning.push_str(&text);
                        on_delta(Delta::Reasoning(&text));
                        spoke = true;
                    }
                }
            }
            for call in delta["tool_calls"].as_array().into_iter().flatten() {
                // Calls arrive split across deltas and are keyed by index, not by order:
                // the name lands in the first delta, the arguments in the ones after.
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
                        // Set here and applied below: `slot` holds a borrow of `self.calls`
                        // for the length of this loop.
                        spoke = true;
                    }
                }
            }
        }
        if spoke {
            self.first_token.get_or_insert_with(Instant::now);
        }
    }

    /// Everything up to a `</think>` was thinking, however it arrived; what follows the
    /// terminator is where the answer starts.
    fn cut_thought(&mut self, buffer: &str) -> Option<String> {
        let at = buffer.find(THOUGHT_END)?;
        self.reasoning
            .push_str(&buffer[..at].replace("<think>", ""));
        Some(buffer[at + THOUGHT_END.len()..].to_string())
    }

    /// Returns whether any of it reached the reader. Text held back against a `</think>`
    /// that may still arrive has not.
    fn answer(&mut self, text: String, on_delta: &mut dyn FnMut(Delta<'_>)) -> bool {
        // Only a response that has already streamed reasoning can still have its answer
        // turn out to be the tail of a thought.
        if self.settled || self.reasoning.is_empty() {
            self.text.push_str(&text);
            on_delta(Delta::Text(&text));
            return true;
        }
        self.withheld.push_str(&text);
        let held = std::mem::take(&mut self.withheld);
        let ready = match self.cut_thought(&held) {
            Some(tail) => tail,
            None if held.len() > HOLD => held,
            None => {
                self.withheld = held;
                return false;
            }
        };
        self.settled = true;
        if ready.is_empty() {
            return false;
        }
        self.text.push_str(&ready);
        on_delta(Delta::Text(&ready));
        true
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
        if self.calls.is_empty() {
            self.text.push_str(&held);
            on_delta(Delta::Text(&held));
        } else {
            self.reasoning.push_str(&held);
            on_delta(Delta::Reasoning(&held));
        }
        self.first_token.get_or_insert_with(Instant::now);
    }

    /// Always after `flush`, which is what places whatever was held back.
    fn finish(mut self, sent: Instant) -> Completion {
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
            ttft_ms: self
                .first_token
                .map(|at| at.duration_since(sent).as_millis() as u64),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_gateways_own_name_for_a_model_finds_the_weights_it_serves() {
        assert_eq!(known_window("nemotron-3-ultra"), 262_144);
        assert_eq!(
            known_window("NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"),
            262_144
        );
        assert_eq!(known_window("grok-2"), 131_072);
        // A nickname that sits inside a longer released name still finds it, because
        // every match agrees on the window.
        assert_eq!(known_window("nemotron-3"), 262_144);
        // Weights nobody here has a figure for stay unknown rather than guessed.
        assert_eq!(known_window("gpt-4o"), 0);
        assert_eq!(known_window("grok"), 0);
        assert_eq!(known_window(""), 0);
    }
}

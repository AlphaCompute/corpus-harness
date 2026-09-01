//! Anthropic messages, written and read as though they were OpenAI completions.
//!
//! The accumulator in `lib.rs` is where thinking is split from answer, tool arguments are
//! re-assembled and control tokens are stripped, and none of that is protocol-specific.
//! So this module does not accumulate anything: it turns one Anthropic stream event into
//! the OpenAI chunks that same accumulator already knows how to read, and the rest of the
//! crate never learns there was a second protocol.
//!
//! The request direction is a rearrangement rather than a translation. A system message
//! moves out of the list into its own slot, a tool result stops being a role of its own
//! and becomes a content block on the user turn that answers the call, and every result
//! for one turn has to arrive on a single such turn or the provider rejects a
//! conversation that was perfectly well formed in OpenAI's shape.

use std::collections::BTreeMap;

use serde_json::{Value, json};

use crate::{Message, Role, Tool};

/// Anthropic's floor for extended thinking; a budget under it is refused outright.
pub const MIN_THINKING_BUDGET: u32 = 1024;

/// One request, in the shape Anthropic reads.
pub fn request(
    model: &str,
    messages: &[Message],
    tools: &[Tool],
    max_tokens: u32,
    thinking_budget: Option<u32>,
    stream: bool,
) -> Value {
    let mut system = Vec::new();
    let mut turns: Vec<Value> = Vec::new();
    for message in messages {
        match message.role {
            Role::System => {
                if let Some(text) = message.content.as_deref().filter(|t| !t.is_empty()) {
                    system.push(text.to_string());
                }
            }
            Role::Tool => {
                let block = json!({
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id.clone().unwrap_or_default(),
                    "content": message.content.clone().unwrap_or_default(),
                });
                // Folded onto the previous turn when that turn is itself nothing but
                // results: OpenAI sends one message per result, Anthropic wants them
                // together on the one user turn that answers the assistant's calls.
                match turns.last_mut().filter(|last| is_results_turn(last)) {
                    Some(last) => {
                        if let Some(content) = last["content"].as_array_mut() {
                            content.push(block);
                        }
                    }
                    None => turns.push(json!({"role": "user", "content": [block]})),
                }
            }
            Role::Assistant => {
                let mut blocks = Vec::new();
                if let Some(text) = message.content.as_deref().filter(|t| !t.is_empty()) {
                    blocks.push(json!({"type": "text", "text": text}));
                }
                for call in &message.tool_calls {
                    blocks.push(json!({
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        // The model wrote these and the provider passed them through, so
                        // they are what the conversation contains. Anthropic wants an
                        // object; unparsable text is carried under a key rather than
                        // dropping the call out of the history entirely.
                        "input": serde_json::from_str::<Value>(&call.arguments)
                            .ok()
                            .filter(Value::is_object)
                            .unwrap_or_else(|| json!({"_unparsed": call.arguments})),
                    }));
                }
                if !blocks.is_empty() {
                    turns.push(json!({"role": "assistant", "content": blocks}));
                }
            }
            Role::User => {
                if let Some(text) = message.content.as_deref().filter(|t| !t.is_empty()) {
                    turns.push(json!({
                        "role": "user",
                        "content": [{"type": "text", "text": text}],
                    }));
                }
            }
        }
    }

    let mut body = json!({
        "model": model,
        "messages": turns,
        "max_tokens": max_tokens,
        "stream": stream,
    });
    if !system.is_empty() {
        body["system"] = json!(system.join("\n\n"));
    }
    if !tools.is_empty() {
        body["tools"] = Value::Array(
            tools
                .iter()
                .map(|tool| {
                    json!({
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.parameters,
                    })
                })
                .collect(),
        );
    }
    if let Some(budget) = thinking_budget {
        let budget = budget.max(MIN_THINKING_BUDGET);
        // The ceiling has to clear the thinking budget or the pair is refused, and a
        // caller that named a small ceiling was sizing the answer, not the thinking.
        body["max_tokens"] = json!(max_tokens.max(budget + 1024));
        body["thinking"] = json!({"type": "enabled", "budget_tokens": budget});
    }
    body
}

fn is_results_turn(turn: &Value) -> bool {
    turn["role"] == "user"
        && turn["content"]
            .as_array()
            .is_some_and(|blocks| blocks.iter().all(|block| block["type"] == "tool_result"))
}

/// Anthropic's typed block events, re-told as the OpenAI chunks the accumulator reads.
///
/// The two protocols disagree about what a stream is made of: Anthropic opens and closes
/// typed blocks and says which one each delta belongs to, while OpenAI has one delta
/// channel where text, reasoning and tool arguments are told apart by which field is
/// present. So the open block has to be remembered, and tool calls have to be numbered in
/// the order they were opened — OpenAI correlates argument fragments by an ordinal of its
/// own, not by Anthropic's block index.
#[derive(Default)]
pub struct Stream {
    tool_ordinals: BTreeMap<u64, u64>,
    input_tokens: u64,
    output_tokens: u64,
    cache_read: Option<u64>,
}

impl Stream {
    /// The chunks one event becomes, and whether the stream has announced its end.
    pub fn feed(&mut self, event: &Value) -> (Vec<Value>, bool) {
        let mut out = Vec::new();
        match event["type"].as_str().unwrap_or_default() {
            "message_start" => {
                self.read_usage(&event["message"]["usage"]);
                out.push(delta(json!({"role": "assistant"}), None, None));
            }
            "content_block_start" => {
                let index = event["index"].as_u64().unwrap_or(0);
                if event["content_block"]["type"] == "tool_use" {
                    let ordinal = self.tool_ordinals.len() as u64;
                    self.tool_ordinals.insert(index, ordinal);
                    out.push(delta(
                        json!({"tool_calls": [{
                            "index": ordinal,
                            "id": event["content_block"]["id"],
                            "type": "function",
                            "function": {
                                "name": event["content_block"]["name"],
                                "arguments": "",
                            },
                        }]}),
                        None,
                        None,
                    ));
                }
            }
            "content_block_delta" => {
                let index = event["index"].as_u64().unwrap_or(0);
                match event["delta"]["type"].as_str().unwrap_or_default() {
                    "text_delta" => {
                        if let Some(text) = nonempty(&event["delta"]["text"]) {
                            out.push(delta(json!({"content": text}), None, None));
                        }
                    }
                    "thinking_delta" => {
                        if let Some(text) = nonempty(&event["delta"]["thinking"]) {
                            out.push(delta(json!({"reasoning": text}), None, None));
                        }
                    }
                    "input_json_delta" => {
                        if let (Some(fragment), Some(ordinal)) = (
                            nonempty(&event["delta"]["partial_json"]),
                            self.tool_ordinals.get(&index),
                        ) {
                            out.push(delta(
                                json!({"tool_calls": [{
                                    "index": ordinal,
                                    "function": {"arguments": fragment},
                                }]}),
                                None,
                                None,
                            ));
                        }
                    }
                    // signature_delta attests the thinking block. Anthropic-only, with no
                    // OpenAI field to land in, so it is dropped rather than invented into
                    // one.
                    _ => {}
                }
            }
            "message_delta" => {
                self.read_usage(&event["usage"]);
                if let Some(reason) = event["delta"]["stop_reason"].as_str() {
                    out.push(delta(json!({}), Some(finish_reason(reason)), self.usage()));
                }
            }
            "message_stop" => return (out, true),
            _ => {}
        }
        (out, false)
    }

    fn read_usage(&mut self, usage: &Value) {
        // Merged field by field, never wholesale: input_tokens is stated once, in
        // message_start, and a later delta carrying only the output count would zero the
        // prompt half if it replaced the whole record.
        if let Some(input) = usage["input_tokens"].as_u64() {
            self.input_tokens = input;
        }
        if let Some(output) = usage["output_tokens"].as_u64() {
            self.output_tokens = output;
        }
        if let Some(cached) = usage["cache_read_input_tokens"].as_u64() {
            self.cache_read = Some(cached);
        }
    }

    fn usage(&self) -> Option<Value> {
        let mut usage = json!({
            "prompt_tokens": self.input_tokens,
            "completion_tokens": self.output_tokens,
        });
        if let Some(cached) = self.cache_read {
            usage["prompt_tokens_details"] = json!({"cached_tokens": cached});
        }
        Some(usage)
    }
}

fn nonempty(value: &Value) -> Option<&str> {
    value.as_str().filter(|text| !text.is_empty())
}

fn finish_reason(reason: &str) -> &'static str {
    match reason {
        "tool_use" => "tool_calls",
        "max_tokens" => "length",
        _ => "stop",
    }
}

fn delta(delta: Value, finish: Option<&str>, usage: Option<Value>) -> Value {
    let mut chunk = json!({
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    });
    if let Some(usage) = usage {
        chunk["usage"] = usage;
    }
    chunk
}

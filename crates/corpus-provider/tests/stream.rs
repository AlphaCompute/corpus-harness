use corpus_provider::{Completion, Delta, Jsonl, Message, Provider, Role, StopReason, Tool};
use corpus_testkit::{delta, finish, refused, serve, sse, truncated};

async fn ask(base_url: &str) -> (Completion, String) {
    let provider = Provider::new(base_url, "test-key", "test-model");
    let mut seen = String::new();
    let completion = provider
        .stream(
            &[Message::text(Role::User, "hi")],
            &[Tool {
                name: "python".into(),
                description: "run code".into(),
                parameters: serde_json::json!({ "type": "object" }),
            }],
            &mut |d| {
                if let Delta::Text(text) = d {
                    seen.push_str(text)
                }
            },
        )
        .await
        .unwrap();
    (completion, seen)
}

#[tokio::test]
async fn text_deltas_become_one_answer() {
    let body = sse(&[
        &delta(r#""content":"Hel""#),
        &delta(r#""content":"lo""#),
        r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":11,"completion_tokens":2}}"#,
    ]);
    let (completion, streamed) = ask(&serve(vec![body]).await.url).await;

    assert_eq!(completion.text, "Hello");
    assert_eq!(
        streamed, "Hello",
        "deltas must reach the caller as they arrive"
    );
    assert_eq!(completion.stop, StopReason::Stop);
    assert_eq!(completion.usage.input, 11);
    assert_eq!(completion.usage.output, 2);
}

#[tokio::test]
async fn tool_calls_are_assembled_by_index() {
    let body = sse(&[
        &delta(
            r#""tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"python","arguments":""}}]"#,
        ),
        &delta(
            r#""tool_calls":[{"index":1,"id":"call_b","type":"function","function":{"name":"python","arguments":"{\"code\":"}}]"#,
        ),
        &delta(r#""tool_calls":[{"index":0,"function":{"arguments":"{\"code\":\"1+1\"}"}}]"#),
        &delta(r#""tool_calls":[{"index":1,"function":{"arguments":"\"2+2\"}"}}]"#),
        &finish("tool_calls"),
    ]);
    let (completion, _) = ask(&serve(vec![body]).await.url).await;

    assert_eq!(completion.stop, StopReason::ToolCalls);
    let names: Vec<_> = completion
        .tool_calls
        .iter()
        .map(|c| c.name.as_str())
        .collect();
    assert_eq!(names, ["python", "python"]);
    assert_eq!(completion.tool_calls[0].id, "call_a");
    assert_eq!(completion.tool_calls[0].arguments, r#"{"code":"1+1"}"#);
    assert_eq!(completion.tool_calls[1].id, "call_b");
    assert_eq!(completion.tool_calls[1].arguments, r#"{"code":"2+2"}"#);
}

#[tokio::test]
async fn arguments_survive_a_fragment_split_inside_an_escape() {
    let body = sse(&[
        &delta(
            r#""tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"python","arguments":"{\"code\": \"import re"}}]"#,
        ),
        // The escape that opens a new line is split down the middle by the provider.
        &delta(r#""tool_calls":[{"index":0,"function":{"arguments":"\\"}}]"#),
        &delta(
            r#""tool_calls":[{"index":0,"function":{"arguments":"nprint(\\\"\\u043f\\u0440\\u0438\\u0432\\u0435\\u0442\\\")\"}"}}]"#,
        ),
        &finish("tool_calls"),
    ]);
    let provider = Provider::new(&serve(vec![body]).await.url, "test-key", "test-model");
    let mut written = String::new();
    let completion = provider
        .stream(&[Message::text(Role::User, "hi")], &[], &mut |d| {
            if let Delta::Code(code) = d {
                written.push_str(code)
            }
        })
        .await
        .unwrap();

    assert_eq!(
        written, "import re\nprint(\"привет\")",
        "the cell must reach the screen as it is typed, not when it lands"
    );
    let arguments: serde_json::Value =
        serde_json::from_str(&completion.tool_calls[0].arguments).expect("arguments stay valid");
    assert_eq!(
        arguments["code"], written,
        "and what was read on screen must be the cell that then runs"
    );
}

/// The trace is what gets attached to a report about the server, so it has to hold what
/// the server said, not what this crate made of it: the control token that never reaches
/// the answer is still on the wire, and the trace is where it stays visible.
#[tokio::test]
async fn the_wire_trace_keeps_what_the_stream_actually_carried() {
    let body = sse(&[
        &delta(r#""content":"Done.<|im_end|>""#),
        &delta(
            r#""tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"python","arguments":"{\"code\":\"1\"}"}}]"#,
        ),
        &finish("tool_calls"),
    ]);
    let path = std::env::temp_dir().join(format!("corpus-trace-{}.jsonl", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let provider = Provider::new(&serve(vec![body]).await.url, "test-key", "test-model")
        .tracing_to(std::sync::Arc::new(Jsonl::to(&path).unwrap()));
    let completion = provider
        .stream(&[Message::text(Role::User, "hi")], &[], &mut |_| {})
        .await
        .unwrap();

    let trace: Vec<serde_json::Value> = std::fs::read_to_string(&path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let _ = std::fs::remove_file(&path);

    assert!(
        trace.iter().all(|entry| entry["at"].is_string()),
        "every line is stamped: {trace:#?}"
    );
    assert_eq!(trace[0]["post"]["model"], "test-model");
    assert_eq!(trace[1]["status"], 200);
    let wire: String = trace
        .iter()
        .filter_map(|entry| entry["line"].as_str())
        .collect();
    assert!(
        wire.contains("<|im_end|>"),
        "the trace must keep what we strip on the way in: {wire}"
    );
    assert!(
        wire.contains("[DONE]") && wire.contains(r#"\"code\":\"1\""#),
        "and every raw line that carried the call: {wire}"
    );
    assert!(
        !completion.text.contains("<|im_end|>"),
        "while the answer itself stays cleaned"
    );
    assert!(
        trace.iter().filter_map(|e| e["read"].as_u64()).count() > 0,
        "each line says which network read it came off: {trace:#?}"
    );
}

#[tokio::test]
async fn an_answer_that_arrives_only_as_reasoning_is_not_lost() {
    let body = sse(&[
        &delta(r#""reasoning":"the answer ""#),
        &delta(r#""reasoning_content":"is 42""#),
        &finish("stop"),
    ]);
    let (completion, _) = ask(&serve(vec![body]).await.url).await;

    assert_eq!(completion.reasoning, "the answer is 42");
    assert_eq!(completion.text, "the answer is 42");
}

#[tokio::test]
async fn a_broken_stream_is_retried() {
    let good = sse(&[&delta(r#""content":"whole answer""#), &finish("stop")]);
    let base_url = serve(vec![truncated(&[&delta(r#""content":"half""#)]), good])
        .await
        .url;

    let (completion, _) = ask(&base_url).await;
    assert_eq!(completion.text, "whole answer");
}

#[tokio::test]
async fn a_rejected_request_reports_what_the_provider_said() {
    let body = refused("401 Unauthorized", r#"{"error":"bad api key"}"#);
    let provider = Provider::new(serve(vec![body]).await.url, "nope", "test-model");
    let error = provider
        .stream(&[Message::text(Role::User, "hi")], &[], &mut |_| {})
        .await
        .expect_err("4xx must not be swallowed");

    assert!(error.to_string().contains("401"), "{error}");
    assert!(error.to_string().contains("bad api key"), "{error}");
}

#[tokio::test]
async fn the_model_list_comes_back_in_the_order_the_provider_gave_it() {
    let endpoint = corpus_testkit::serve(vec![corpus_testkit::json(
        r#"{"object":"list","data":[
            {"id":"grok-2","context_length":131072},
            {"id":"nemotron-3","max_model_len":8192},
            {"id":"kimi-k3","context_window":262144},
            {"id":"glm-5","limit":{"context":204800,"output":131072}},
            {"id":"nemotron-3-ultra"},
            {"id":"gpt-4o"}
        ]}"#,
    )])
    .await;
    let provider = Provider::new(&endpoint.url, "test-key", "");

    let listed: Vec<(String, u32)> = provider
        .models()
        .await
        .unwrap()
        .into_iter()
        .map(|model| (model.id, model.window))
        .collect();
    assert_eq!(
        listed,
        [
            ("grok-2".to_string(), 131072),
            ("nemotron-3".to_string(), 8192),
            // Every gateway spells the window its own way, and some nest it.
            ("kimi-k3".to_string(), 262144),
            ("glm-5".to_string(), 204800),
            // A gateway that publishes nothing but serves a model we know the spec of.
            ("nemotron-3-ultra".to_string(), 262144),
            // Anything else stays at zero rather than guessed.
            ("gpt-4o".to_string(), 0),
        ]
    );
}

#[tokio::test]
async fn control_tokens_never_reach_the_answer_even_split_across_deltas() {
    let body = sse(&[
        &delta(r#""content":"Hello there.<|sep""#),
        &delta(r#""content":"arator|>""#),
        &delta(r#""content":" And more.<|im_end|>""#),
        &finish("stop"),
    ]);
    let (completion, streamed) = ask(&serve(vec![body]).await.url).await;

    assert_eq!(completion.text, "Hello there. And more.");
    assert_eq!(
        streamed, "Hello there. And more.",
        "and they must not flash on screen either"
    );
}

#[tokio::test]
async fn a_control_token_that_never_closes_is_dropped_not_shown() {
    let body = sse(&[&delta(r#""content":"Done.<|separa""#), &finish("length")]);
    let (completion, _) = ask(&serve(vec![body]).await.url).await;

    assert_eq!(completion.text, "Done.");
}

#[tokio::test]
async fn the_request_carries_no_stop_strings() {
    let endpoint = serve(vec![sse(&[&delta(r#""content":"hi""#)])]).await;
    ask(&endpoint.url).await;

    assert!(
        !endpoint.requests()[0].contains(r#""stop""#),
        "this gateway mangles a stream when stop strings are set"
    );
}

#[tokio::test]
async fn thinking_that_leaks_into_the_answer_is_put_back_where_it_belongs() {
    let body = sse(&[
        &delta(r#""reasoning":"The user wants 71*93. ""#),
        &delta(r#""content":" and so it is 6603.</thi""#),
        &delta(r#""content":"nk>6603""#),
        &finish("stop"),
    ]);
    let (completion, _) = ask(&serve(vec![body]).await.url).await;

    assert_eq!(
        completion.text, "6603",
        "only what follows the terminator is the answer"
    );
    assert!(
        completion.reasoning.contains("The user wants"),
        "{}",
        completion.reasoning
    );
    assert!(
        completion.reasoning.contains("and so it is 6603."),
        "{}",
        completion.reasoning
    );
}

#[tokio::test]
async fn leaked_thinking_never_reaches_the_screen_either() {
    let body = sse(&[
        &delta(r#""reasoning":"Working on it. ""#),
        &delta(r#""content":" and so it is 6603.</thi""#),
        &delta(r#""content":"nk>6603""#),
        &finish("stop"),
    ]);
    let (completion, streamed) = ask(&serve(vec![body]).await.url).await;

    assert_eq!(completion.text, "6603");
    assert_eq!(
        streamed, "6603",
        "held back until it was clear which half it was"
    );
}

#[tokio::test]
async fn an_ordinary_answer_after_thinking_is_not_held_forever() {
    let long = "x".repeat(400);
    let body = sse(&[
        &delta(r#""reasoning":"Thinking. ""#),
        &delta(&format!(r#""content":"{long}""#)),
        &finish("stop"),
    ]);
    let (completion, streamed) = ask(&serve(vec![body]).await.url).await;

    assert_eq!(completion.text, long);
    assert_eq!(
        streamed, long,
        "no terminator ever came, so it must still be shown"
    );
}

#[tokio::test]
async fn a_short_answer_after_thinking_is_still_shown_when_the_stream_ends() {
    let body = sse(&[
        &delta(r#""reasoning":"Thinking about it. ""#),
        &delta(r#""content":"e""#),
        &finish("stop"),
    ]);
    let (completion, streamed) = ask(&serve(vec![body]).await.url).await;

    assert_eq!(completion.text, "e");
    assert_eq!(
        streamed, "e",
        "text in the log that was never on screen is a bug"
    );
}

#[tokio::test]
async fn a_stray_scrap_of_content_beside_a_tool_call_is_not_an_answer() {
    let body = sse(&[
        &delta(r#""reasoning":"I should compute this with the tool.""#),
        &delta(r#""content":"a""#),
        &delta(
            r#""tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"python","arguments":"{}"}}]"#,
        ),
        &finish("tool_calls"),
    ]);
    let (completion, streamed) = ask(&serve(vec![body]).await.url).await;

    assert_eq!(completion.tool_calls.len(), 1);
    assert!(
        completion.text.is_empty(),
        "a model that calls a tool is not answering"
    );
    assert!(
        streamed.is_empty(),
        "and nothing of it should show as answer text"
    );
    assert!(
        completion.reasoning.ends_with('a'),
        "{}",
        completion.reasoning
    );
}

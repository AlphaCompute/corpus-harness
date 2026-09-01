//! The route table, and what a route that speaks the other protocol changes.
//!
//! The failures worth catching here are quiet ones. A vendor slash read as a route sends
//! every `z-ai/glm-5.3` to a provider that does not exist; an Anthropic route reached with
//! an OpenAI body is a 400 that reads like a bad key; and tool-argument fragments numbered
//! by Anthropic's block index assemble into the wrong call the moment a turn makes two.

use corpus_provider::{Delta, Message, Protocol, Provider, Registry, Role, Tool};
use corpus_testkit::{serve, sse};

const TABLE: &str = r#"{
  "providers": {
    "redpill": {
      "displayName": "RedPill",
      "baseURL": "https://api.redpill.ai/v1",
      "models": [
        {"id": "z-ai/glm-5.3", "contextWindow": 1048576,
         "reasoningEfforts": {"off": null, "high": "high"}},
        {"id": "openai/gpt-oss-120b", "maxTokens": 8192}
      ]
    },
    "claude": {
      "api": "anthropic-messages",
      "baseURL": "https://api.anthropic.com/v1",
      "defaultMaxTokens": 2048,
      "models": [
        {"id": "claude-x", "reasoningEfforts": {"off": null, "high": "8000"}}
      ]
    }
  }
}"#;

fn registry() -> Registry {
    Registry::parse(TABLE, "https://gateway.example/v1", "gateway-key").unwrap()
}

#[test]
fn a_vendor_slash_is_not_a_route() {
    let registry = registry();
    let (route, model) = registry.resolve("redpill/z-ai/glm-5.3").unwrap();
    assert_eq!(route.provider, "redpill");
    assert_eq!(model, "z-ai/glm-5.3", "the model keeps its own slash");
}

#[test]
fn an_unknown_prefix_is_a_model_name_not_an_error() {
    // `z-ai` names no route, so the whole string is a model the default endpoint serves.
    // Reading it as a missing provider would fail a session that is perfectly fine.
    let registry = registry();
    let (route, model) = registry.resolve("z-ai/glm-5.3").unwrap();
    assert_eq!(route.provider, corpus_provider::FALLBACK);
    assert_eq!(model, "z-ai/glm-5.3");
}

#[test]
fn an_unprefixed_model_goes_to_the_endpoint_the_process_was_started_with() {
    let registry = registry();
    let (route, model) = registry.resolve("nemotron-3-ultra").unwrap();
    assert_eq!(route.base_url, "https://gateway.example/v1");
    assert_eq!(route.api_key, "gateway-key");
    assert_eq!(model, "nemotron-3-ultra");
}

#[test]
fn a_route_declares_its_own_protocol_and_ceilings() {
    let registry = registry();
    let claude = registry.route("claude").unwrap();
    assert_eq!(claude.protocol, Protocol::AnthropicMessages);
    assert_eq!(claude.max_tokens_for("claude-x"), 2048);
    let redpill = registry.route("redpill").unwrap();
    assert_eq!(redpill.protocol, Protocol::OpenAiCompletions);
    assert_eq!(redpill.max_tokens_for("openai/gpt-oss-120b"), 8192);
    assert_eq!(
        redpill.max_tokens_for("a-model-it-never-listed"),
        corpus_provider::DEFAULT_MAX_TOKENS
    );
}

#[test]
fn a_credential_is_read_from_the_variable_the_table_names() {
    // Named per test: the environment is process-wide and tests share it.
    unsafe { std::env::set_var("ROUTES_TEST_KEY_A", "sk-from-env") };
    let table = r#"{"providers":{"p":{"baseURL":"http://x/v1","apiKeyEnv":"ROUTES_TEST_KEY_A"}}}"#;
    let registry = Registry::parse(table, "http://d/v1", "d").unwrap();
    assert_eq!(registry.route("p").unwrap().api_key, "sk-from-env");
}

#[test]
fn a_named_but_unset_credential_fails_where_it_is_written() {
    let table =
        r#"{"providers":{"p":{"baseURL":"http://x/v1","apiKeyEnv":"ROUTES_TEST_KEY_UNSET"}}}"#;
    let refused = format!("{:#}", Registry::parse(table, "http://d/v1", "d").unwrap_err());
    assert!(
        refused.contains("ROUTES_TEST_KEY_UNSET"),
        "the refusal must name the variable: {refused}"
    );
}

#[test]
fn an_unknown_protocol_is_refused() {
    let table = r#"{"providers":{"p":{"api":"grpc","baseURL":"http://x/v1"}}}"#;
    // `{:#}` renders the whole chain: the route it names is the outer context, and the
    // protocol it could not read is the cause underneath it.
    let refused = format!("{:#}", Registry::parse(table, "http://d/v1", "d").unwrap_err());
    assert!(refused.contains("grpc"), "{refused}");
}

#[test]
fn a_route_key_may_not_carry_a_slash() {
    let table = r#"{"providers":{"a/b":{"baseURL":"http://x/v1"}}}"#;
    let refused = format!("{:#}", Registry::parse(table, "http://d/v1", "d").unwrap_err());
    assert!(refused.contains("slash"), "{refused}");
}

#[test]
fn only_a_routed_model_is_qualified_when_it_is_named_back() {
    let registry = registry();
    assert_eq!(registry.qualify("redpill", "z-ai/glm-5.3"), "redpill/z-ai/glm-5.3");
    assert_eq!(
        registry.qualify(corpus_provider::FALLBACK, "nemotron-3-ultra"),
        "nemotron-3-ultra"
    );
}

#[test]
fn an_unoffered_reasoning_level_is_refused_before_a_request_is_built() {
    let registry = registry();
    let route = registry.route("redpill").unwrap();
    // `.err()`, not `.unwrap_err()`: the Ok half holds an http client and a trace file,
    // neither of which is worth a Debug impl for one assertion.
    let refused = Provider::on(route, "z-ai/glm-5.3")
        .reasoning(route, "max")
        .err()
        .expect("an unoffered level must be refused")
        .to_string();
    assert!(refused.contains("max"), "{refused}");
    assert!(
        refused.contains("high"),
        "it should say what is offered: {refused}"
    );
    assert!(Provider::on(route, "z-ai/glm-5.3").reasoning(route, "high").is_ok());
}

async fn ask(provider: Provider, messages: &[Message]) -> (String, String, Vec<String>) {
    let mut text = String::new();
    let mut reasoning = String::new();
    let completion = provider
        .stream(
            messages,
            &[Tool {
                name: "python".into(),
                description: "run code".into(),
                parameters: serde_json::json!({"type": "object"}),
            }],
            &mut |d| match d {
                Delta::Text(chunk) => text.push_str(chunk),
                Delta::Reasoning(chunk) => reasoning.push_str(chunk),
                Delta::Code(_) => {}
            },
        )
        .await
        .unwrap();
    let calls = completion
        .tool_calls
        .iter()
        .map(|call| format!("{}:{}", call.name, call.arguments))
        .collect();
    assert_eq!(completion.text, text, "the answer is what was streamed");
    (text, reasoning, calls)
}

#[tokio::test]
async fn an_anthropic_route_is_dialled_in_its_own_dialect() {
    let body = sse(&[
        r#"{"type":"message_start","message":{"id":"m","usage":{"input_tokens":9}}}"#,
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"text"}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}"#,
        r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}"#,
        r#"{"type":"message_stop"}"#,
    ]);
    let endpoint = serve(vec![body]).await;
    let table = format!(
        r#"{{"providers":{{"claude":{{"api":"anthropic-messages","baseURL":"{}","defaultMaxTokens":2048}}}}}}"#,
        endpoint.url
    );
    let registry = Registry::parse(&table, "http://unused/v1", "unused").unwrap();
    let (route, model) = registry.resolve("claude/claude-x").unwrap();

    let (text, _, _) = ask(
        Provider::on(route, &model),
        &[
            Message::text(Role::System, "be brief"),
            Message::text(Role::User, "hi"),
        ],
    )
    .await;

    assert_eq!(text, "Hello", "the answer survived translation");

    let head = &endpoint.heads()[0];
    assert!(head.starts_with("POST /messages"), "dialled: {head}");
    assert!(head.contains("x-api-key"), "auth scheme: {head}");
    assert!(head.contains("anthropic-version"), "version header: {head}");
    assert!(
        !head.to_lowercase().contains("authorization:"),
        "no bearer token belongs on this route: {head}"
    );

    let sent: serde_json::Value = serde_json::from_str(&endpoint.requests()[0]).unwrap();
    assert_eq!(sent["system"], "be brief", "the system message moved slots");
    assert_eq!(sent["max_tokens"], 2048, "a ceiling is always supplied");
    assert_eq!(sent["messages"][0]["role"], "user");
    assert_eq!(sent["tools"][0]["input_schema"]["type"], "object");
}

#[tokio::test]
async fn anthropic_tool_calls_keep_their_own_arguments() {
    let body = sse(&[
        r#"{"type":"message_start","message":{"id":"m","usage":{}}}"#,
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"a","name":"python"}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"code\":"}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"1+1\"}"}}"#,
        r#"{"type":"content_block_stop","index":0}"#,
        r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"b","name":"python"}}"#,
        r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"code\":\"2+2\"}"}}"#,
        r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"}}"#,
        r#"{"type":"message_stop"}"#,
    ]);
    let endpoint = serve(vec![body]).await;
    let table = format!(
        r#"{{"providers":{{"claude":{{"api":"anthropic-messages","baseURL":"{}"}}}}}}"#,
        endpoint.url
    );
    let registry = Registry::parse(&table, "http://unused/v1", "unused").unwrap();
    let (route, model) = registry.resolve("claude/claude-x").unwrap();

    let (_, _, calls) = ask(
        Provider::on(route, &model),
        &[Message::text(Role::User, "add")],
    )
    .await;

    assert_eq!(
        calls,
        vec![
            "python:{\"code\":\"1+1\"}".to_string(),
            "python:{\"code\":\"2+2\"}".to_string()
        ],
        "each call must keep the fragments that belong to it"
    );
}

#[tokio::test]
async fn anthropic_thinking_is_not_mistaken_for_the_answer() {
    let body = sse(&[
        r#"{"type":"message_start","message":{"id":"m","usage":{}}}"#,
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking"}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"weighing"}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig"}}"#,
        r#"{"type":"content_block_start","index":1,"content_block":{"type":"text"}}"#,
        r#"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"done"}}"#,
        r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":4}}"#,
        r#"{"type":"message_stop"}"#,
    ]);
    let endpoint = serve(vec![body]).await;
    let table = format!(
        r#"{{"providers":{{"claude":{{"api":"anthropic-messages","baseURL":"{}"}}}}}}"#,
        endpoint.url
    );
    let registry = Registry::parse(&table, "http://unused/v1", "unused").unwrap();
    let (route, model) = registry.resolve("claude/claude-x").unwrap();

    let (text, reasoning, _) = ask(
        Provider::on(route, &model),
        &[Message::text(Role::User, "think")],
    )
    .await;

    assert_eq!(text, "done");
    assert_eq!(reasoning, "weighing");
}

#[tokio::test]
async fn a_reasoning_level_travels_as_its_route_spells_it() {
    let openai = serve(vec![sse(&[
        r#"{"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":null}]}"#,
        r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
    ])])
    .await;
    let anthropic = serve(vec![sse(&[
        r#"{"type":"message_start","message":{"id":"m","usage":{}}}"#,
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"text"}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}"#,
        r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#,
        r#"{"type":"message_stop"}"#,
    ])])
    .await;
    let table = format!(
        r#"{{"providers":{{
             "rp":{{"baseURL":"{}","models":[{{"id":"m","reasoningEfforts":{{"high":"high"}}}}]}},
             "cl":{{"api":"anthropic-messages","baseURL":"{}","defaultMaxTokens":1000,
                    "models":[{{"id":"m","reasoningEfforts":{{"high":"8000"}}}}]}}
           }}}}"#,
        openai.url, anthropic.url
    );
    let registry = Registry::parse(&table, "http://unused/v1", "unused").unwrap();

    let (route, model) = registry.resolve("rp/m").unwrap();
    let provider = Provider::on(route, &model).reasoning(route, "high").unwrap();
    ask(provider, &[Message::text(Role::User, "hi")]).await;
    let sent: serde_json::Value = serde_json::from_str(&openai.requests()[0]).unwrap();
    assert_eq!(
        sent["reasoning_effort"], "high",
        "an OpenAI route carries the level as a parameter"
    );

    let (route, model) = registry.resolve("cl/m").unwrap();
    let provider = Provider::on(route, &model).reasoning(route, "high").unwrap();
    ask(provider, &[Message::text(Role::User, "hi")]).await;
    let sent: serde_json::Value = serde_json::from_str(&anthropic.requests()[0]).unwrap();
    assert_eq!(
        sent["thinking"]["budget_tokens"], 8000,
        "an Anthropic route spells the level as a token budget"
    );
    assert!(
        sent["max_tokens"].as_u64().unwrap() > 8000,
        "the ceiling must clear the budget: {sent}"
    );
}

#[tokio::test]
async fn a_tool_result_becomes_a_content_block_on_one_user_turn() {
    let body = sse(&[
        r#"{"type":"message_start","message":{"id":"m","usage":{}}}"#,
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"text"}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}"#,
        r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#,
        r#"{"type":"message_stop"}"#,
    ]);
    let endpoint = serve(vec![body]).await;
    let table = format!(
        r#"{{"providers":{{"cl":{{"api":"anthropic-messages","baseURL":"{}"}}}}}}"#,
        endpoint.url
    );
    let registry = Registry::parse(&table, "http://unused/v1", "unused").unwrap();
    let (route, model) = registry.resolve("cl/claude-x").unwrap();

    ask(
        Provider::on(route, &model),
        &[
            Message::text(Role::User, "go"),
            Message::calls(
                String::new(),
                vec![
                    corpus_provider::ToolCall {
                        id: "a".into(),
                        name: "python".into(),
                        arguments: r#"{"code":"1"}"#.into(),
                    },
                    corpus_provider::ToolCall {
                        id: "b".into(),
                        name: "python".into(),
                        arguments: r#"{"code":"2"}"#.into(),
                    },
                ],
            ),
            Message::tool_result("a", "one"),
            Message::tool_result("b", "two"),
        ],
    )
    .await;

    let sent: serde_json::Value = serde_json::from_str(&endpoint.requests()[0]).unwrap();
    let turns = sent["messages"].as_array().unwrap();
    assert_eq!(turns.len(), 3, "the two results share one turn: {sent}");
    let results = turns[2]["content"].as_array().unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0]["tool_use_id"], "a");
    assert_eq!(results[1]["tool_use_id"], "b");
    assert_eq!(turns[1]["content"][0]["type"], "tool_use");
    assert_eq!(turns[1]["content"][0]["input"]["code"], "1");
}

#[tokio::test]
async fn an_anthropic_stream_that_is_cut_still_yields_what_arrived() {
    // No message_stop: the provider hung up mid-answer. What was streamed is still the
    // best account of the turn, and a session that discards it has lost work it paid for.
    let body = sse(&[
        r#"{"type":"message_start","message":{"id":"m","usage":{}}}"#,
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"text"}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"half"}}"#,
    ]);
    let endpoint = serve(vec![body]).await;
    let table = format!(
        r#"{{"providers":{{"cl":{{"api":"anthropic-messages","baseURL":"{}"}}}}}}"#,
        endpoint.url
    );
    let registry = Registry::parse(&table, "http://unused/v1", "unused").unwrap();
    let (route, model) = registry.resolve("cl/claude-x").unwrap();

    let (text, _, _) = ask(
        Provider::on(route, &model),
        &[Message::text(Role::User, "hi")],
    )
    .await;
    assert_eq!(text, "half");
}

#[test]
fn a_table_may_repoint_the_default_without_every_caller_naming_a_route() {
    let table = r#"{"providers":{"default":{"baseURL":"http://bench:9/v1"}}}"#;
    let registry = Registry::parse(table, "https://gateway.example/v1", "k").unwrap();
    let (route, _) = registry.resolve("some-model").unwrap();
    assert_eq!(route.base_url, "http://bench:9/v1");
}

#[test]
fn a_level_travels_as_written_when_the_route_describes_no_such_model() {
    // The ordinary case behind a router: this process holds the endpoint, and the
    // catalogue that knows which levels exist lives on the other side of it. Refusing
    // here would make the level unusable exactly where it is meant to be configured.
    let table = r#"{"providers":{"p":{"baseURL":"http://x/v1"}}}"#;
    let registry = Registry::parse(table, "http://d/v1", "d").unwrap();
    let route = registry.route("p").unwrap();
    assert!(
        Provider::on(route, "a-model-it-never-listed")
            .reasoning(route, "high")
            .is_ok()
    );
}

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use corpus_agent::{Agent, Budget, Event, Outcome};
use corpus_kernel::{Host, Kernel};
use corpus_provider::{Provider, StopReason};
use corpus_testkit::{
    Endpoint, delta, finish, kernel_dir, python_call, runs_python, says, serve, sse,
};
use serde_json::{Value, json};

#[derive(Default)]
struct Recorder {
    seen: Mutex<Vec<String>>,
}

#[async_trait]
impl Host for Recorder {
    async fn call(&self, name: &str, args: Value) -> Result<Value, String> {
        self.seen.lock().unwrap().push(name.to_string());
        Ok(json!({ "results": [format!("a page about {}", args["query"])] }))
    }
}

async fn agent(endpoint: &Endpoint, host: Arc<Recorder>, budget: Budget) -> Agent {
    let kernel = Kernel::start("python3", &kernel_dir(), &["web_search"])
        .await
        .unwrap();
    Agent::new(
        Provider::new(&endpoint.url, "test-key", "test-model"),
        kernel,
        host,
        "You are Corpus.",
        budget,
    )
}

async fn ask(agent: &mut Agent, prompt: &str) -> (Outcome, Vec<Event>) {
    let mut events = Vec::new();
    let outcome = agent
        .run(prompt, &mut |event| events.push(event))
        .await
        .unwrap();
    (outcome, events)
}

#[tokio::test]
async fn an_answer_from_knowledge_costs_one_model_call() {
    let endpoint = serve(vec![says("Paris.")]).await;
    let mut agent = agent(&endpoint, Arc::new(Recorder::default()), Budget::default()).await;

    let (outcome, events) = ask(&mut agent, "Capital of France?").await;

    assert_eq!(outcome.text, "Paris.");
    assert_eq!(outcome.stop, StopReason::Stop);
    assert_eq!(endpoint.requests().len(), 1);
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Event::Answer { text, .. } if text == "Paris."))
    );
    assert!(!events.iter().any(|e| matches!(e, Event::ToolStart { .. })));
}

#[tokio::test]
async fn a_question_that_needs_looking_up_runs_tools_and_finishes() {
    let endpoint = serve(vec![
        runs_python(
            "call_1",
            "hits = web_search(query='corpus')\nprint(hits['results'][0])",
        ),
        says("Found it."),
    ])
    .await;
    let host = Arc::new(Recorder::default());
    let mut agent = agent(&endpoint, host.clone(), Budget::default()).await;

    let (outcome, events) = ask(&mut agent, "Look up corpus.").await;

    assert_eq!(outcome.text, "Found it.");
    assert_eq!(host.seen.lock().unwrap().as_slice(), ["web_search"]);
    let streamed: String = events
        .iter()
        .filter_map(|e| match e {
            Event::ToolStream { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        streamed.contains("a page about"),
        "cell output must stream live: {streamed:?}"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Event::ToolEnd { ok: true, .. }))
    );
}

#[tokio::test]
async fn running_out_of_steps_yields_a_partial_answer_not_a_crash() {
    let endpoint = serve(vec![
        runs_python("call_1", "print('one')"),
        runs_python("call_2", "print('two')"),
    ])
    .await;
    let budget = Budget {
        max_steps: 2,
        ..Budget::default()
    };
    let mut agent = agent(&endpoint, Arc::new(Recorder::default()), budget).await;

    let (outcome, events) = ask(&mut agent, "Keep going forever.").await;

    assert_eq!(outcome.stop, StopReason::Partial);
    assert!(
        !outcome.text.is_empty(),
        "a partial answer is still an answer"
    );
    assert_eq!(endpoint.requests().len(), 2);
    assert!(events.iter().any(|e| matches!(
        e,
        Event::TurnEnd {
            stop: StopReason::Partial,
            ..
        }
    )));
}

#[tokio::test]
async fn a_broken_tool_call_is_data_for_the_model_not_a_dead_turn() {
    let endpoint = serve(vec![
        runs_python("call_1", "this is not python("),
        says("Right, my syntax was off."),
    ])
    .await;
    let mut agent = agent(&endpoint, Arc::new(Recorder::default()), Budget::default()).await;

    let (outcome, events) = ask(&mut agent, "Run something broken.").await;

    assert_eq!(outcome.text, "Right, my syntax was off.");
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Event::ToolEnd { ok: false, .. }))
    );
    assert!(
        endpoint.requests()[1].contains("SyntaxError"),
        "the model must see why the cell failed"
    );
}

#[tokio::test]
async fn compaction_drops_old_tool_output_and_keeps_the_question() {
    let big = "x = 'a' * 200_000\nprint(x)";
    let endpoint = serve(vec![
        runs_python("call_1", big),
        runs_python("call_2", "print('second')"),
        says("Summarised."),
    ])
    .await;
    let budget = Budget {
        context_chars: 5_000,
        ..Budget::default()
    };
    let mut agent = agent(&endpoint, Arc::new(Recorder::default()), budget).await;

    let (outcome, events) = ask(&mut agent, "Fetch something enormous.").await;

    assert_eq!(outcome.text, "Summarised.");
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Event::Compaction { dropped, .. } if *dropped > 0))
    );
    let last = endpoint.requests().last().unwrap().clone();
    assert!(
        last.contains("output dropped"),
        "old tool output should be gone"
    );
    assert!(
        last.contains("Fetch something enormous"),
        "the question must survive compaction"
    );
}

#[tokio::test]
async fn a_resumed_transcript_does_not_promise_a_namespace_that_is_gone() {
    let endpoint = serve(vec![says("It is gone; I will fetch it again.")]).await;
    let mut agent = agent(&endpoint, Arc::new(Recorder::default()), Budget::default()).await;

    agent.replay([Event::UserMessage {
        turn_id: uuid::Uuid::nil(),
        agent: uuid::Uuid::nil(),
        text: "Fetch the page into `page`.".into(),
    }]);
    ask(&mut agent, "What was in it?").await;

    let sent = &endpoint.requests()[0];
    assert!(
        sent.contains("interpreter is fresh"),
        "a replayed session must say its variables did not come back: {sent}"
    );
}

/// A log holds every agent's stream, and only one of them is the session. The root cannot
/// be named outright — this process minted a fresh id — so it is whoever no `AgentStart`
/// announced, and the child's dialogue must stay out of the context that comes back.
#[tokio::test]
async fn a_replayed_log_brings_back_the_root_and_not_its_children() {
    let endpoint = serve(vec![says("Both, still.")]).await;
    let mut agent = agent(&endpoint, Arc::new(Recorder::default()), Budget::default()).await;
    let root = uuid::Uuid::now_v7();
    let child = uuid::Uuid::now_v7();
    let turn = uuid::Uuid::now_v7();

    agent.replay([
        Event::UserMessage {
            turn_id: turn,
            agent: root,
            text: "Read both reports.".into(),
        },
        Event::AgentStart {
            agent: child,
            parent: root,
            task: "Read the second report.".into(),
        },
        Event::UserMessage {
            turn_id: uuid::Uuid::now_v7(),
            agent: child,
            text: "Read the second report.".into(),
        },
        Event::Answer {
            agent: child,
            text: "The second report says margins fell.".into(),
        },
        Event::AgentEnd {
            agent: child,
            parent: root,
            ok: true,
            chars: 36,
            preview: "The second report says margins fell.".into(),
        },
        Event::Answer {
            agent: root,
            text: "Both are read.".into(),
        },
        Event::Notice {
            turn_id: uuid::Uuid::now_v7(),
            agent: root,
            text: "an agent finished: 36 chars".into(),
        },
    ]);
    ask(&mut agent, "What did they say?").await;

    let sent = &endpoint.requests()[0];
    assert!(sent.contains("Read both reports."), "{sent}");
    assert!(sent.contains("Both are read."), "{sent}");
    assert!(
        !sent.contains("margins fell"),
        "the child's own dialogue is not the session's: {sent}"
    );
    assert!(
        sent.contains("an agent finished") && sent.contains("not the person"),
        "a turn nobody typed must come back saying so: {sent}"
    );
}

#[tokio::test]
async fn a_result_too_big_for_the_context_points_at_the_variable() {
    let endpoint = serve(vec![
        runs_python("call_1", "print('x' * 20_000)"),
        says("Sliced it."),
    ])
    .await;
    let mut agent = agent(&endpoint, Arc::new(Recorder::default()), Budget::default()).await;

    let (outcome, _) = ask(&mut agent, "Make something enormous.").await;

    assert_eq!(outcome.text, "Sliced it.");
    assert!(
        endpoint.requests()[1].contains("whole in `_`"),
        "a cut result must tell the model where the value still is"
    );
}

#[tokio::test]
async fn a_cell_that_never_returns_does_not_hang_the_turn() {
    let endpoint = serve(vec![runs_python("call_1", "while True: pass")]).await;
    let budget = Budget {
        cell_timeout: Duration::from_millis(300),
        max_steps: 1,
        ..Budget::default()
    };
    let mut agent = agent(&endpoint, Arc::new(Recorder::default()), budget).await;

    let (outcome, events) = ask(&mut agent, "Loop forever.").await;

    assert_eq!(outcome.stop, StopReason::Partial);
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Event::ToolEnd { ok: false, .. }))
    );
}

#[tokio::test]
async fn an_interrupt_ends_the_turn_with_whatever_it_has() {
    let endpoint = serve(vec![runs_python(
        "call_1",
        "print('working')\nimport time\ntime.sleep(30)",
    )])
    .await;
    let mut agent = agent(&endpoint, Arc::new(Recorder::default()), Budget::default()).await;
    let interrupt = agent.interrupter();

    let mut events = Vec::new();
    let outcome = agent
        .run("take your time", &mut |event| {
            if matches!(event, Event::ToolStream { .. }) {
                interrupt.raise();
            }
            events.push(event);
        })
        .await
        .unwrap();

    assert_eq!(outcome.stop, StopReason::Partial);
    assert!(
        !outcome.text.is_empty(),
        "an interrupted turn still answers something"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Event::ToolEnd { ok: false, .. }))
    );
    assert_eq!(
        endpoint.requests().len(),
        1,
        "the loop must not start another step"
    );
}

#[tokio::test]
async fn narration_next_to_a_tool_call_is_not_mistaken_for_an_answer() {
    let call = python_call("call_1", "print('one')");
    let calling = sse(&[
        &delta(&format!(
            r#""content":"Let me look.","tool_calls":[{call}]"#
        )),
        &finish("tool_calls"),
    ]);
    let endpoint = serve(vec![calling.clone(), calling]).await;
    let budget = Budget {
        max_steps: 2,
        ..Budget::default()
    };
    let mut agent = agent(&endpoint, Arc::new(Recorder::default()), budget).await;

    let (outcome, _) = ask(&mut agent, "Keep going.").await;

    assert_eq!(outcome.stop, StopReason::Partial);
    assert!(
        outcome.text.contains("Stopped after"),
        "a half-finished chain answers with what happened, not with its own narration: {}",
        outcome.text
    );
}

#[tokio::test]
async fn a_model_that_stops_answering_does_not_hold_the_turn_forever() {
    let endpoint = serve(vec![Vec::new()]).await;
    let budget = Budget {
        wall_clock: Duration::from_millis(400),
        ..Budget::default()
    };
    let mut agent = agent(&endpoint, Arc::new(Recorder::default()), budget).await;

    let started = std::time::Instant::now();
    let (outcome, _) = ask(&mut agent, "hello?").await;

    assert_eq!(outcome.stop, StopReason::Partial);
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "the clock has to run during the request, not only between them"
    );
}

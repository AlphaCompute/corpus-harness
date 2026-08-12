//! Agents a session sends off, over the whole path: the cell that spawns one, the child's
//! own loop, and the answer coming back into the cell that asked for it.

use std::path::{Path, PathBuf};

use corpus_provider::Jsonl;
use corpus_testkit::{Endpoint, kernel_dir, runs_python, says, serve};
use serde_json::Value;

const BIN: &str = env!("CARGO_BIN_EXE_corpus");

fn workdir(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

async fn corpus(endpoint: &Endpoint, prompt: &str, log: &Path) {
    let output = tokio::process::Command::new(BIN)
        .env("CORPUS_BASE_URL", &endpoint.url)
        .env("CORPUS_API_KEY", "test-key")
        .env("CORPUS_MODEL", "test-model")
        .env("CORPUS_KERNEL_DIR", kernel_dir())
        .args([prompt, "--log", log.to_str().unwrap()])
        .output()
        .await
        .unwrap();
    assert!(
        output.status.success(),
        "corpus failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn events(log: &Path) -> Vec<Value> {
    Jsonl::read::<Value>(log).unwrap()
}

fn printed(log: &Path) -> String {
    events(log)
        .iter()
        .filter(|event| event["t"] == "tool_stream")
        .map(|event| event["text"].as_str().unwrap_or_default().to_string())
        .collect()
}

/// One cell sends an agent off and waits for it in a loop, which is the shape the prompt
/// asks for. Three model calls: the cell, the child's turn, and the answer that follows.
#[tokio::test]
async fn a_cell_sends_an_agent_off_and_reads_what_it_answered() {
    let dir = workdir("spawn");
    let log = dir.join("session.jsonl");
    let code = "kid = spawn('read the second report')\n\
                answer = None\n\
                while answer is None:\n\
                \x20   answer = kid.result(timeout=10)\n\
                print('the agent said:', answer)\n\
                print('and it is done:', kid.done())\n";
    let endpoint = serve(vec![
        runs_python("call_1", code),
        says("Margins fell by a fifth."),
        says("The second report is read."),
    ])
    .await;

    corpus(&endpoint, "Read the second report.", &log).await;

    let printed = printed(&log);
    assert!(
        printed.contains("Margins fell by a fifth."),
        "the child's answer never reached the cell that asked: {printed}"
    );
    assert!(
        printed.contains("UNTRUSTED CONTENT"),
        "an agent reports on what it read, so its report is material too: {printed}"
    );
    assert!(printed.contains("and it is done: True"), "{printed}");

    let log = events(&log);
    let started = log
        .iter()
        .find(|event| event["t"] == "agent_start")
        .expect("the log says a child was sent off");
    let ended = log
        .iter()
        .find(|event| event["t"] == "agent_end")
        .expect("the log says it came back");
    assert_eq!(started["task"], "read the second report");
    assert_eq!(ended["ok"], true);
    assert_eq!(ended["chars"], 24);
    assert_eq!(ended["preview"], "Margins fell by a fifth.");
    assert_eq!(
        started["agent"], ended["agent"],
        "one child, one identity, from the line that starts it to the line that ends it"
    );
    assert!(
        log.iter().any(
            |event| event["t"] == "answer" && event["agent"] == started["agent"]
                && event["text"] == "Margins fell by a fifth."
        ),
        "the child's own stream belongs in the log under its own name"
    );
}

/// Depth is the list of names a namespace was bound from, so this is the whole of the
/// rule: a child has everything its parent has except the power to make another child.
#[tokio::test]
async fn a_child_cannot_send_off_a_child_of_its_own() {
    let dir = workdir("depth");
    let log = dir.join("session.jsonl");
    let parent = "kid = spawn('say what you have')\n\
                  while (answer := kid.result(timeout=10)) is None:\n\
                  \x20   pass\n\
                  print(answer)\n";
    let child = "print('mine:', sorted(n for n in dir() if n in ('spawn', 'agents', 'llm_batch', 'fetch_url')))";
    let endpoint = serve(vec![
        runs_python("call_1", parent),
        runs_python("call_2", child),
        says("I have fetch_url and llm_batch."),
        says("It has no agents of its own."),
    ])
    .await;

    corpus(&endpoint, "What can the agent you send do?", &log).await;

    let printed = printed(&log);
    assert!(
        printed.contains("mine: ['fetch_url', 'llm_batch']"),
        "a child's namespace must hold the tools and not the delegation: {printed}"
    );
}

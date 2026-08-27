//! Scaffolding shared by the CLI's integration tests: how the binary is launched, where it
//! runs, and how its log is read back afterwards.
//!
//! This lives in the test tree rather than in `corpus-testkit` because the two things it is
//! built on are given to a test target and to nothing else: `CARGO_BIN_EXE_corpus` names the
//! binary cargo just built, and `CARGO_TARGET_TMPDIR` names a scratch directory for the
//! crate's tests. Both resolve here, in each test binary that declares `mod common`.

// Each test binary compiles its own copy of this module and uses a different part of it.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::Output;

use corpus_provider::Jsonl;
use corpus_testkit::{Endpoint, kernel_dir};
use serde_json::Value;

pub const BIN: &str = env!("CARGO_BIN_EXE_corpus");

/// A directory of its own per test, emptied first: a rerun must not read what the last run
/// left behind.
pub fn workdir(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The binary under test, pointed at the scripted endpoint and the kernel in the tree. Said
/// once, so a variable the binary comes to need is added here and nowhere else.
pub fn command(endpoint: &Endpoint) -> tokio::process::Command {
    let mut command = tokio::process::Command::new(BIN);
    command
        .env("CORPUS_BASE_URL", &endpoint.url)
        .env("CORPUS_API_KEY", "test-key")
        .env("CORPUS_MODEL", "test-model")
        .env("CORPUS_KERNEL_DIR", kernel_dir());
    command
}

/// Runs a command the caller has already configured and insists the process was happy,
/// because a test reading the log of a run that crashed reads an empty file and says
/// something confusing about it. Separate from [`corpus`] for the tests that need a
/// working directory or a `HOME` of their own, which is not something an argument list
/// can carry.
pub async fn succeeds(command: &mut tokio::process::Command) -> Output {
    let args: Vec<String> = command
        .as_std()
        .get_args()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect();
    let output = command.output().await.unwrap();
    assert!(
        output.status.success(),
        "corpus {args:?} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

/// Runs to completion and insists the process was happy.
pub async fn corpus(endpoint: &Endpoint, args: &[&str], env: &[(&str, &str)]) -> Output {
    let mut command = command(endpoint);
    command.args(args);
    for (name, value) in env {
        command.env(name, value);
    }
    succeeds(&mut command).await
}

/// One prompt, logged where the test can read it: the shape most of these tests want.
pub async fn ask(endpoint: &Endpoint, prompt: &str, log: &Path) -> Output {
    corpus(endpoint, &[prompt, "--log", log.to_str().unwrap()], &[]).await
}

/// A served session with both pipes open. Dropping it kills the child, so a test that fails
/// early does not leave one behind.
pub struct Served {
    _child: tokio::process::Child,
    pub stdin: tokio::process::ChildStdin,
    pub lines: tokio::io::Lines<tokio::io::BufReader<tokio::process::ChildStdout>>,
}

/// Starts `corpus serve` and hands back the two ends of its protocol. What each test does
/// with them differs; getting here is what they all have in common.
pub fn served(endpoint: &Endpoint) -> Served {
    use tokio::io::{AsyncBufReadExt, BufReader};

    let mut child = command(endpoint)
        .arg("serve")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let stdin = child.stdin.take().unwrap();
    let lines = BufReader::new(child.stdout.take().unwrap()).lines();
    Served {
        _child: child,
        stdin,
        lines,
    }
}

/// A line of the protocol a served session reads on stdin. Built rather than spelled out,
/// so a prompt with a quote in it does not have to be escaped by hand at the call site and
/// cannot quietly stop being valid json.
fn run_line(text: &str) -> Vec<u8> {
    let mut line = serde_json::json!({ "cmd": "run", "text": text }).to_string();
    line.push('\n');
    line.into_bytes()
}

const INTERRUPT_LINE: &[u8] = b"{\"cmd\":\"interrupt\"}\n";

/// The longest a served turn may take before the test gives up. A test that hangs on the
/// pipe says nothing at all about why it hung.
const TURN: std::time::Duration = std::time::Duration::from_secs(60);

/// What a test wants done about the event it was just handed.
pub enum Step {
    /// Keep reading.
    Go,
    /// Send an interrupt down the pipe, then keep reading.
    Interrupt,
    /// The event this turn was being read for.
    Stop,
}

impl Served {
    /// One turn, as a client reads it: the prompt goes down the pipe, every event that
    /// comes back is handed to `read`, and what it says to do about each one is what
    /// happens. Gives back the events it read, the one it stopped on last.
    ///
    /// Which ending to stop on is the caller's, and never simply the first: every agent
    /// over there shares this one pipe, so a child finishing crosses it too. The served
    /// client reads the same shape for the same reason — see `Remote::pump`.
    pub async fn turn(&mut self, prompt: &str, read: impl FnMut(&Value) -> Step) -> Vec<Value> {
        tokio::time::timeout(TURN, self.follow(prompt, read))
            .await
            .expect("the served session never finished the turn")
    }

    async fn follow(&mut self, prompt: &str, mut read: impl FnMut(&Value) -> Step) -> Vec<Value> {
        use tokio::io::AsyncWriteExt;

        self.stdin.write_all(&run_line(prompt)).await.unwrap();
        let mut seen = Vec::new();
        while let Some(line) = self.lines.next_line().await.unwrap() {
            let event: Value = serde_json::from_str(&line).unwrap();
            let step = read(&event);
            seen.push(event);
            match step {
                Step::Interrupt => self.stdin.write_all(INTERRUPT_LINE).await.unwrap(),
                Step::Stop => return seen,
                Step::Go => {}
            }
        }
        panic!("the served session ended without finishing the turn");
    }
}

/// Whether this is the end of a turn, which is where most of these tests stop reading.
pub fn ends_turn(event: &Value) -> Step {
    match event["t"] == "turn_end" {
        true => Step::Stop,
        false => Step::Go,
    }
}

/// The log as it was written.
pub fn events(log: &Path) -> Vec<Value> {
    Jsonl::read::<Value>(log).unwrap()
}

/// Identifiers and timestamps are minted per run, so a transcript is only comparable
/// without them.
pub fn transcript(log: &Path) -> Vec<Value> {
    events(log)
        .into_iter()
        .map(|mut event| {
            // Ids and clocks: what a second run of the same session cannot repeat, and
            // what no assertion about a transcript is ever about.
            for key in [
                "session_id",
                "turn_id",
                "agent",
                "at",
                "llm_ms",
                "ttft_ms",
                "wall_ms",
            ] {
                event.as_object_mut().unwrap().remove(key);
            }
            event
        })
        .collect()
}

/// Every `key` carried by the events of one `kind`, in the order the log holds them.
/// `.concat()` puts a streamed field back together; a `Vec` keeps one event's value apart
/// from the next one's.
pub fn field(log: &Path, kind: &str, key: &str) -> Vec<String> {
    transcript(log)
        .iter()
        .filter(|event| event["t"] == kind)
        .map(|event| event[key].as_str().unwrap_or_default().to_string())
        .collect()
}

/// Everything the cells printed, put back together as the session saw it.
pub fn printed(log: &Path) -> String {
    field(log, "tool_stream", "text").concat()
}

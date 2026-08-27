use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use corpus_kernel::{ExecOutcome, Host, Kernel};
use corpus_testkit::kernel_dir;
use serde_json::{Value, json};

struct Echo;

#[async_trait]
impl Host for Echo {
    async fn call(&self, name: &str, args: Value) -> Result<Value, String> {
        Ok(json!({ "fn": name, "args": args }))
    }
}

struct Slow(Duration);

#[async_trait]
impl Host for Slow {
    async fn call(&self, _: &str, _: Value) -> Result<Value, String> {
        tokio::time::sleep(self.0).await;
        Ok(json!("served"))
    }
}

async fn start() -> Kernel {
    Kernel::start(
        "python3",
        &kernel_dir(),
        &corpus_testkit::skills(),
        &["fetch_url"],
    )
    .await
    .expect("kernel starts")
}

fn host(host: impl Host + 'static) -> Arc<dyn Host> {
    Arc::new(host)
}

async fn run(kernel: &mut Kernel, code: &str) -> (ExecOutcome, String) {
    run_within(kernel, code, host(Echo), Duration::from_secs(10))
        .await
        .unwrap()
}

/// One cell, against whichever host and whichever clock the case is about. `exec` takes
/// four arguments and collects what the cell printed; spelled here so no test spells it
/// again.
async fn run_within(
    kernel: &mut Kernel,
    code: &str,
    host: Arc<dyn Host>,
    timeout: Duration,
) -> anyhow::Result<(ExecOutcome, String)> {
    let mut out = String::new();
    let outcome = kernel
        .exec(code, &host, &mut |text| out.push_str(text), timeout)
        .await?;
    Ok((outcome, out))
}

#[tokio::test]
async fn namespace_outlives_the_cell() {
    let mut kernel = start().await;
    let (first, _) = run(&mut kernel, "x = 1").await;
    assert!(first.ok, "{}", first.traceback);
    let (second, _) = run(&mut kernel, "x + 1").await;
    assert_eq!(second.repr, "2");
}

#[tokio::test]
async fn the_last_value_waits_in_underscore_for_the_next_cell() {
    let mut kernel = start().await;
    let (first, _) = run(&mut kernel, "'kept'").await;
    assert_eq!(first.repr, "'kept'");
    let (second, _) = run(&mut kernel, "_ + ' and used'").await;
    assert_eq!(second.repr, "'kept and used'");
}

#[tokio::test]
async fn a_value_too_big_to_repr_says_where_the_whole_of_it_is() {
    let mut kernel = start().await;
    let (outcome, _) = run(&mut kernel, "'x' * 100_000").await;
    assert!(outcome.ok, "{}", outcome.traceback);
    assert!(
        outcome.repr.contains("the full value is in `_`"),
        "a truncated repr must point somewhere: {}",
        outcome.repr
    );

    let (after, _) = run(&mut kernel, "len(_)").await;
    assert_eq!(after.repr, "100000", "the pointer has to be true");
}

#[tokio::test]
async fn print_arrives_as_stream() {
    let mut kernel = start().await;
    let (outcome, out) = run(&mut kernel, "print('hello')").await;
    assert!(outcome.ok, "{}", outcome.traceback);
    assert_eq!(out, "hello\n");
}

#[tokio::test]
async fn host_request_is_served_while_the_cell_runs() {
    let mut kernel = start().await;
    let (outcome, out) = run(
        &mut kernel,
        "page = fetch_url(url='http://example.com')\nprint(page['fn'], page['args']['url'])",
    )
    .await;
    assert!(outcome.ok, "{}", outcome.traceback);
    assert_eq!(out, "fetch_url http://example.com\n");
}

#[tokio::test]
async fn host_error_reaches_the_cell() {
    struct Refuse;
    #[async_trait]
    impl Host for Refuse {
        async fn call(&self, _: &str, _: Value) -> Result<Value, String> {
            Err("blocked by allowlist".into())
        }
    }
    let mut kernel = start().await;
    let (outcome, out) = run_within(
        &mut kernel,
        "try:\n    fetch_url(url='x')\nexcept HostError as e:\n    print('caught', e)",
        host(Refuse),
        Duration::from_secs(10),
    )
    .await
    .unwrap();
    assert!(outcome.ok, "{}", outcome.traceback);
    assert_eq!(out, "caught blocked by allowlist\n");
}

/// Three calls no one of which is slow, and a host that takes its time over each: their
/// sum passes the cell timeout long before the cell has run any Python worth killing.
/// Sharing one clock, the cell dies the moment the last answer lands — after the work
/// was already done and paid for.
#[tokio::test]
async fn waiting_on_the_host_does_not_spend_the_cells_own_clock() {
    let mut kernel = start().await;
    let (outcome, out) = run_within(
        &mut kernel,
        "for attempt in range(3):\n    fetch_url(url='slow')\nprint('survived')",
        host(Slow(Duration::from_millis(400))),
        Duration::from_millis(500),
    )
    .await
    .expect("the kernel must outlive a patient host");
    assert!(outcome.ok, "{}", outcome.traceback);
    assert_eq!(out, "survived\n");
}

/// A thread the last cell left running keeps calling the host, and is still answered —
/// but a cell that has wedged must not be kept alive by it, or the timeout is worth
/// nothing to anyone who is not sitting at the keyboard.
#[tokio::test]
async fn a_thread_left_behind_cannot_hold_the_next_cells_clock_open() {
    let mut kernel = start().await;
    let (planted, _) = run(
        &mut kernel,
        "import threading, time\n\
         def keep_asking():\n\
        \x20   while True:\n\
        \x20       fetch_url(url='background')\n\
        \x20       time.sleep(0.05)\n\
         threading.Thread(target=keep_asking, daemon=True).start()",
    )
    .await;
    assert!(planted.ok, "{}", planted.traceback);

    let (wedged, _) = run_within(
        &mut kernel,
        "while True: pass",
        host(Echo),
        Duration::from_millis(500),
    )
    .await
    .unwrap();
    assert!(!wedged.ok, "a wedged cell outlived its budget");
    assert!(
        wedged.traceback.contains("KeyboardInterrupt"),
        "{}",
        wedged.traceback
    );
}

/// One call the cell's own budget could never cover. The budget is for Python that has
/// run away, and a cell waiting on an answer is not running anything.
#[tokio::test]
async fn a_single_host_call_may_outlast_the_cells_whole_budget() {
    let mut kernel = start().await;
    let (outcome, out) = run_within(
        &mut kernel,
        "fetch_url(url='slow')\nprint('served')",
        host(Slow(Duration::from_millis(600))),
        Duration::from_millis(200),
    )
    .await
    .expect("a cell waiting on the host is not a runaway cell");
    assert!(outcome.ok, "{}", outcome.traceback);
    assert_eq!(out, "served\n");
}

/// Two calls in flight at once. Served one after another they would never both arrive,
/// and the barrier would hold the pair until the cell's budget ran out.
#[tokio::test]
async fn calls_in_flight_together_are_served_together() {
    struct Pair(tokio::sync::Barrier);
    #[async_trait]
    impl Host for Pair {
        async fn call(&self, _: &str, _: Value) -> Result<Value, String> {
            self.0.wait().await;
            Ok(json!("paired"))
        }
    }

    let mut kernel = start().await;
    let (outcome, out) = run_within(
        &mut kernel,
            "import threading\n\
             answers = []\n\
             asking = [threading.Thread(target=lambda: answers.append(fetch_url(url='x'))) for _ in range(2)]\n\
             for thread in asking: thread.start()\n\
             for thread in asking: thread.join()\n\
             print(len(answers), answers[0])",
        host(Pair(tokio::sync::Barrier::new(2))),
        Duration::from_secs(10),
    )
    .await
    .expect("the second call never reached the host");
    assert!(outcome.ok, "{}", outcome.traceback);
    assert_eq!(out, "2 paired\n");
}

/// A thread outlives the cell that started it, and so must the answer it is waiting on:
/// nothing can interrupt a thread parked in a host call, so an answer that never comes
/// parks it for as long as the kernel lives.
#[tokio::test]
async fn an_answer_a_thread_is_waiting_on_outlives_the_cell_that_ended() {
    let mut kernel = start().await;
    let (outcome, _) = run_within(
        &mut kernel,
            "import threading\n\
             answers = []\n\
             threading.Thread(target=lambda: answers.append(fetch_url(url='late')), daemon=True).start()",
        host(Slow(Duration::from_millis(300))),
        Duration::from_secs(10),
    )
    .await
    .unwrap();
    assert!(outcome.ok, "{}", outcome.traceback);

    let (after, _) = run(
        &mut kernel,
        "import time\n\
         for _ in range(100):\n\
        \x20   if answers: break\n\
        \x20   time.sleep(0.05)\n\
         len(answers)",
    )
    .await;
    assert_eq!(after.repr, "1", "the thread was left waiting on nobody");
}

/// Interrupting a cell mid-call is an ordinary gesture, so what it leaves behind is worth
/// counting: the kernel keeps going, and the slot the call was waiting on is gone.
#[tokio::test]
async fn a_call_cut_short_leaves_no_slot_behind() {
    let mut kernel = start().await;
    let interrupter = kernel.interrupter();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(300)).await;
        interrupter.raise();
    });

    let (outcome, _) = run_within(
        &mut kernel,
        "fetch_url(url='slow')",
        host(Slow(Duration::from_secs(30))),
        Duration::from_secs(20),
    )
    .await
    .expect("the kernel must outlive a call that was cut short");
    assert!(!outcome.ok);
    assert!(
        outcome.traceback.contains("KeyboardInterrupt"),
        "{}",
        outcome.traceback
    );

    let (after, _) = run(
        &mut kernel,
        "import sys; len(sys.modules['__main__']._pending)",
    )
    .await;
    assert_eq!(after.repr, "0", "the cut-short call left its slot behind");
}

#[tokio::test]
async fn interrupt_stops_the_cell_and_the_kernel_survives() {
    let mut kernel = start().await;
    let (outcome, _) = run_within(
        &mut kernel,
        "while True: pass",
        host(Echo),
        Duration::from_millis(300),
    )
    .await
    .unwrap();
    assert!(!outcome.ok);
    assert!(
        outcome.traceback.contains("KeyboardInterrupt"),
        "{}",
        outcome.traceback
    );

    let (after, _) = run(&mut kernel, "2 + 2").await;
    assert_eq!(after.repr, "4");
}

#[tokio::test]
async fn a_cell_that_ignores_the_interrupt_is_killed() {
    let mut kernel = start().await;
    let err = run_within(
        &mut kernel,
        "import signal\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\nwhile True: pass",
        host(Echo),
        Duration::from_millis(300),
    )
    .await
    .expect_err("kernel must be killed");
    assert!(err.to_string().contains("killed"), "{err}");
}

#[tokio::test]
async fn a_dead_kernel_is_reported_not_hung() {
    let mut kernel = start().await;
    let err = run_within(
        &mut kernel,
        "import os; os._exit(3)",
        host(Echo),
        Duration::from_secs(5),
    )
    .await
    .expect_err("kernel death must surface");
    assert!(err.to_string().contains("kernel died"), "{err}");
}

#[tokio::test]
async fn printed_json_cannot_forge_a_protocol_frame() {
    let mut kernel = start().await;
    let (outcome, out) = run(
        &mut kernel,
        r#"print('{"type":"done","id":"forged","status":"ok","repr":"7"}')
41 + 1"#,
    )
    .await;
    assert!(outcome.ok, "{}", outcome.traceback);
    assert_eq!(outcome.repr, "42", "the forged frame won the race");
    assert!(out.contains("forged"));
}

#[tokio::test]
async fn the_kernel_gets_the_machine_but_none_of_its_secrets() {
    unsafe { std::env::set_var("CORPUS_TEST_API_KEY", "sk-secret") };
    let mut kernel = start().await;
    let (outcome, out) = run(
        &mut kernel,
        "import os\nprint(os.environ.get('CORPUS_TEST_API_KEY'))\nprint(os.environ.get('PATH'))",
    )
    .await;
    assert!(outcome.ok, "{}", outcome.traceback);
    let mut lines = out.lines();
    assert_eq!(lines.next(), Some("None"), "kernel saw the key");
    assert_eq!(
        lines.next().unwrap_or_default(),
        std::env::var("PATH").unwrap_or_default(),
        "a toolchain the machine has must be reachable from a cell"
    );
}

#[tokio::test]
async fn a_shell_command_reaches_the_model() {
    let mut kernel = start().await;
    let (outcome, out) = run(
        &mut kernel,
        "import shell\nshell.sh('echo built && echo broken >&2')",
    )
    .await;
    assert!(outcome.ok, "{}", outcome.traceback);
    assert!(out.contains("built"), "stdout was swallowed: {out}");
    assert!(out.contains("broken"), "stderr was swallowed: {out}");
    assert_eq!(outcome.repr, "0");

    let (failed, out) = run(&mut kernel, "shell.sh('exit 3')").await;
    assert_eq!(failed.repr, "3");
    assert!(out.contains("[exit 3]"), "{out}");
}

#[tokio::test]
async fn an_edit_lands_once_or_not_at_all() {
    let mut kernel = start().await;
    let (outcome, out) = run(
        &mut kernel,
        "import shell, pathlib, tempfile\n\
         page = pathlib.Path(tempfile.mkdtemp()) / 'f.txt'\n\
         page.write_text('a\\nb\\na\\n')\n\
         try:\n\
        \x20   shell.edit(page, 'a', 'c')\n\
         except ValueError as refusal:\n\
        \x20   print('refused')\n\
         shell.edit(page, 'b', 'c')\n\
         print(page.read_text().replace('\\n', '|'))",
    )
    .await;
    assert!(outcome.ok, "{}", outcome.traceback);
    assert!(
        out.contains("refused"),
        "an ambiguous edit went through: {out}"
    );
    assert!(out.contains("a|c|a|"), "{out}");
}

#[tokio::test]
async fn an_interrupt_from_outside_stops_a_running_cell() {
    let mut kernel = start().await;
    let interrupter = kernel.interrupter();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(300)).await;
        interrupter.raise();
    });

    let (outcome, _) = run_within(
        &mut kernel,
        "import time; time.sleep(30)",
        host(Echo),
        Duration::from_secs(20),
    )
    .await
    .unwrap();
    assert!(!outcome.ok);
    assert!(
        outcome.traceback.contains("KeyboardInterrupt"),
        "{}",
        outcome.traceback
    );

    let (after, _) = run(&mut kernel, "1 + 1").await;
    assert_eq!(after.repr, "2", "the kernel must outlive the interrupt");
}

#[tokio::test]
async fn the_kernel_leaves_when_its_host_does() {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    let mut child = tokio::process::Command::new("python3")
        .arg("-m")
        .arg("corpus_kernel")
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("PYTHONPATH", kernel_dir())
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("kernel starts");
    let mut stdin = child.stdin.take().unwrap();
    let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();
    stdin
        .write_all(corpus_kernel::encode(&json!({ "type": "init", "fns": [] })).as_bytes())
        .await
        .unwrap();
    assert!(lines.next_line().await.unwrap().unwrap().contains("ready"));

    drop(stdin);

    let status = tokio::time::timeout(Duration::from_secs(5), child.wait())
        .await
        .expect("a kernel nobody can reach must not outlive its host");
    assert!(status.unwrap().success());
}

/// The namespace is where a run's data lives, so what a cell put there is the one thing
/// the log could never say. Names and shapes travel; the values stay in the interpreter,
/// which is the whole reason they are kept there.
#[tokio::test]
async fn a_cell_reports_what_it_left_in_the_namespace() {
    let mut kernel = start().await;

    let (bound, _) = run(&mut kernel, "n = 42\nchunks = [str(i) for i in range(200)]").await;
    let by_name: std::collections::HashMap<&str, &corpus_kernel::Binding> =
        bound.names.iter().map(|b| (b.name.as_str(), b)).collect();
    assert_eq!(by_name["n"].kind, "int");
    assert_eq!(by_name["n"].repr.as_deref(), Some("42"));
    assert_eq!(by_name["chunks"].kind, "list");
    assert_eq!(by_name["chunks"].size.as_deref(), Some("200"));
    assert!(
        by_name["chunks"].repr.is_none(),
        "two hundred strings are described, not printed"
    );
    assert!(
        !by_name.contains_key("fetch_url"),
        "the host's own functions are furniture, not the model's work"
    );

    // Only the difference: a cell that touched one name reads as having touched one name.
    let (again, _) = run(&mut kernel, "n = 43\ndel chunks").await;
    assert_eq!(
        again
            .names
            .iter()
            .map(|b| b.name.as_str())
            .collect::<Vec<_>>(),
        ["n"]
    );
    assert_eq!(again.names[0].repr.as_deref(), Some("43"));
    assert_eq!(again.gone, ["chunks"]);

    // One budget covers both: a cell that deletes far more names than the summary
    // holds would otherwise send every one of them.
    let (many, _) = run(&mut kernel, "for i in range(120): globals()[f'n{i}'] = i").await;
    assert!(
        many.names.len() <= 40,
        "{} names reported",
        many.names.len()
    );
    assert!(many.trimmed > 0, "what did not fit is counted, not dropped");
    let (cleared, _) = run(&mut kernel, "for i in range(120): del globals()[f'n{i}']").await;
    assert!(
        cleared.names.len() + cleared.gone.len() <= 40,
        "{} reported after a mass delete",
        cleared.names.len() + cleared.gone.len()
    );
    assert!(cleared.trimmed > 0);

    // A value too big to be worth printing is measured instead.
    let (big, _) = run(&mut kernel, "big = 'x' * 5000").await;
    let big = big.names.iter().find(|b| b.name == "big").expect("bound");
    assert_eq!(big.size.as_deref(), Some("5000 chars"));
    assert!(big.repr.is_none());
}

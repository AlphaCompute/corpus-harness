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

async fn start() -> Kernel {
    Kernel::start("python3", &kernel_dir(), &["fetch_url"])
        .await
        .expect("kernel starts")
}

async fn run(kernel: &mut Kernel, code: &str) -> (ExecOutcome, String) {
    run_within(kernel, code, Duration::from_secs(10))
        .await
        .unwrap()
}

async fn run_within(
    kernel: &mut Kernel,
    code: &str,
    timeout: Duration,
) -> anyhow::Result<(ExecOutcome, String)> {
    let mut out = String::new();
    let outcome = kernel
        .exec(code, &Echo, &mut |text| out.push_str(text), timeout)
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
    let mut out = String::new();
    let outcome = kernel
        .exec(
            "try:\n    fetch_url(url='x')\nexcept HostError as e:\n    print('caught', e)",
            &Refuse,
            &mut |text| out.push_str(text),
            Duration::from_secs(10),
        )
        .await
        .unwrap();
    assert!(outcome.ok, "{}", outcome.traceback);
    assert_eq!(out, "caught blocked by allowlist\n");
}

#[tokio::test]
async fn interrupt_stops_the_cell_and_the_kernel_survives() {
    let mut kernel = start().await;
    let (outcome, _) = run_within(&mut kernel, "while True: pass", Duration::from_millis(300))
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
async fn the_kernel_environment_carries_no_secrets() {
    unsafe { std::env::set_var("CORPUS_TEST_API_KEY", "sk-secret") };
    let mut kernel = start().await;
    let (outcome, out) = run(
        &mut kernel,
        "import os\nprint(os.environ.get('CORPUS_TEST_API_KEY'))",
    )
    .await;
    assert!(outcome.ok, "{}", outcome.traceback);
    assert_eq!(out.trim(), "None", "kernel saw the key");
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
        .write_all(b"{\"type\":\"init\",\"fns\":[]}\n")
        .await
        .unwrap();
    assert!(lines.next_line().await.unwrap().unwrap().contains("ready"));

    drop(stdin);

    let status = tokio::time::timeout(Duration::from_secs(5), child.wait())
        .await
        .expect("a kernel nobody can reach must not outlive its host");
    assert!(status.unwrap().success());
}

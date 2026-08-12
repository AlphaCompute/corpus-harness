use std::path::{Path, PathBuf};
use std::process::Output;

use corpus_provider::Jsonl;
use corpus_testkit::{Endpoint, kernel_dir, says, serve};
use serde_json::Value;

const BIN: &str = env!("CARGO_BIN_EXE_corpus");

fn workdir(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The binary under test, pointed at the scripted endpoint and the kernel in the tree.
fn command(endpoint: &Endpoint) -> tokio::process::Command {
    let mut command = tokio::process::Command::new(BIN);
    command
        .env("CORPUS_BASE_URL", &endpoint.url)
        .env("CORPUS_API_KEY", "test-key")
        .env("CORPUS_MODEL", "test-model")
        .env("CORPUS_KERNEL_DIR", kernel_dir());
    command
}

async fn corpus(endpoint: &Endpoint, args: &[&str], env: &[(&str, &str)]) -> Output {
    let mut command = command(endpoint);
    command.args(args);
    for (name, value) in env {
        command.env(name, value);
    }
    let output = command.output().await.unwrap();
    assert!(
        output.status.success(),
        "corpus {args:?} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

/// Identifiers and timestamps are minted per run, so a transcript is only comparable
/// without them.
fn transcript(log: &Path) -> Vec<Value> {
    Jsonl::read::<Value>(log)
        .unwrap()
        .into_iter()
        .map(|mut event| {
            for key in ["session_id", "turn_id", "agent", "at"] {
                event.as_object_mut().unwrap().remove(key);
            }
            event
        })
        .collect()
}

/// Every `key` carried by the events of one `kind`, in the order the log holds them.
/// `.concat()` puts a streamed field back together; a `Vec` keeps one event's value apart
/// from the next one's.
fn field(log: &Path, kind: &str, key: &str) -> Vec<String> {
    transcript(log)
        .iter()
        .filter(|event| event["t"] == kind)
        .map(|event| event[key].as_str().unwrap_or_default().to_string())
        .collect()
}

#[tokio::test]
async fn a_local_run_draws_the_stream_and_writes_the_log() {
    let dir = workdir("local-run");
    let log = dir.join("session.jsonl");
    let endpoint = serve(vec![says("Paris.")]).await;

    let output = corpus(
        &endpoint,
        &["Capital of France?", "--log", log.to_str().unwrap()],
        &[],
    )
    .await;

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Paris."),
        "the answer must reach the terminal as it streams"
    );
    assert!(
        stdout.starts_with(&format!(
            " \\|/\n -*-  corpus v{} · test-model\n /|\\  by ALPHA COMPUTE\n",
            env!("CARGO_PKG_VERSION")
        )),
        "a run opens under the mark, unstyled into a pipe: {stdout}"
    );
    let kinds: Vec<_> = transcript(&log)
        .iter()
        .map(|e| e["t"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(
        kinds,
        [
            "session_start",
            "turn_start",
            "user_message",
            "message_delta",
            "answer",
            "turn_end",
            "session_end"
        ]
    );
}

/// A session leaves nothing on disk unless it was asked to.
#[tokio::test]
async fn the_log_is_written_only_when_asked_for() {
    let dir = workdir("opt-in-log");
    let endpoint = serve(vec![says("Paris."), says("Paris.")]).await;

    let mut quiet = command(&endpoint);
    quiet.arg("Capital of France?").current_dir(&dir);
    assert!(quiet.output().await.unwrap().status.success());
    assert_eq!(
        std::fs::read_dir(&dir).unwrap().count(),
        0,
        "an unasked-for session log was written"
    );

    let logs = dir.join("logs");
    corpus(
        &endpoint,
        &["Capital of France?"],
        &[("CORPUS_LOG", logs.to_str().unwrap())],
    )
    .await;
    let written = std::fs::read_dir(&logs).unwrap().next().unwrap().unwrap();
    assert!(
        !transcript(&written.path()).is_empty(),
        "CORPUS_LOG named a directory and nothing landed in it"
    );
}

#[tokio::test]
async fn resume_continues_the_session_from_its_log() {
    let dir = workdir("resume");
    let log = dir.join("session.jsonl");
    let endpoint = serve(vec![says("Paris."), says("About 2.1 million.")]).await;
    let log_arg = log.to_str().unwrap();

    corpus(&endpoint, &["Capital of France?", "--log", log_arg], &[]).await;
    corpus(
        &endpoint,
        &["And how many people live there?", "--resume", log_arg],
        &[],
    )
    .await;

    let second = &endpoint.requests()[1];
    assert!(
        second.contains("Capital of France"),
        "the earlier question is missing: {second}"
    );
    assert!(
        second.contains("Paris."),
        "the earlier answer is missing: {second}"
    );
    let answers = field(&log, "answer", "text");
    assert_eq!(
        answers,
        ["Paris.", "About 2.1 million."],
        "one log, both turns"
    );
}

#[tokio::test]
async fn a_session_behind_a_pipe_produces_the_same_transcript() {
    let dir = workdir("parity");
    let here = dir.join("local.jsonl");
    let there = dir.join("remote.jsonl");
    let endpoint = serve(vec![says("Paris."), says("Paris.")]).await;

    corpus(
        &endpoint,
        &["Capital of France?", "--log", here.to_str().unwrap()],
        &[],
    )
    .await;
    corpus(
        &endpoint,
        &[
            "connect",
            "Capital of France?",
            "--log",
            there.to_str().unwrap(),
            "--",
            BIN,
            "serve",
        ],
        &[],
    )
    .await;

    assert_eq!(transcript(&here), transcript(&there));
}

#[tokio::test]
async fn an_interrupt_travels_down_the_pipe() {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    let endpoint = serve(vec![corpus_testkit::runs_python(
        "call_1",
        "print('working')\nimport time\ntime.sleep(30)",
    )])
    .await;
    let mut child = command(&endpoint)
        .arg("serve")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();

    let turn = async {
        stdin
            .write_all(b"{\"cmd\":\"run\",\"text\":\"go\"}\n")
            .await
            .unwrap();
        let mut sent = false;
        while let Some(line) = lines.next_line().await.unwrap() {
            let event: Value = serde_json::from_str(&line).unwrap();
            if event["t"] == "tool_stream" && !sent {
                sent = true;
                stdin.write_all(b"{\"cmd\":\"interrupt\"}\n").await.unwrap();
            }
            if event["t"] == "turn_end" {
                return event["stop"].as_str().unwrap_or_default().to_string();
            }
        }
        String::from("the served session ended without a turn_end")
    };

    let stop = tokio::time::timeout(std::time::Duration::from_secs(30), turn)
        .await
        .expect("the interrupt never reached the running cell");
    assert_eq!(stop, "partial");
}

#[tokio::test]
async fn with_no_model_named_the_first_one_offered_is_used() {
    let dir = workdir("first-model");
    let log = dir.join("session.jsonl");
    let endpoint = serve(vec![
        corpus_testkit::json(r#"{"object":"list","data":[{"id":"grok-2"},{"id":"nemotron-3"}]}"#),
        says("Paris."),
    ])
    .await;

    corpus(
        &endpoint,
        &["Capital of France?", "--log", log.to_str().unwrap()],
        &[("CORPUS_MODEL", "")],
    )
    .await;

    let start = transcript(&log).into_iter().next().unwrap();
    assert_eq!(start["model"], "grok-2");
}

/// Documents are only a capability if the packages import inside the kernel as the kernel
/// actually starts it — cleared environment, its own interpreter, nothing inherited. So the
/// cell writes both formats and reads them back, and the round-tripped text is the proof.
#[tokio::test]
async fn the_kernel_writes_and_reads_documents() {
    let dir = workdir("documents");
    let log = dir.join("session.jsonl");
    let pdf = dir.join("report.pdf");
    let docx = dir.join("report.docx");
    let code = format!(
        "from reportlab.pdfgen.canvas import Canvas\n\
         from pypdf import PdfReader\n\
         import corpus_docs, docx\n\
         page = Canvas('{pdf}')\n\
         page.setFont(corpus_docs.unicode_font(), 14)\n\
         page.drawString(72, 720, 'Квартальный отчёт')\n\
         page.save()\n\
         print('pdf says:', PdfReader('{pdf}').pages[0].extract_text().strip())\n\
         paper = docx.Document()\n\
         paper.add_heading('Квартальный отчёт', 0)\n\
         paper.save('{docx}')\n\
         print('docx says:', docx.Document('{docx}').paragraphs[0].text)\n",
        pdf = pdf.display(),
        docx = docx.display(),
    );
    let endpoint = serve(vec![
        corpus_testkit::runs_python("call_1", &code),
        says("Both documents are written."),
    ])
    .await;

    corpus(
        &endpoint,
        &["Write the report.", "--log", log.to_str().unwrap()],
        &[],
    )
    .await;

    let printed = field(&log, "tool_stream", "text").concat();
    assert!(
        printed.contains("pdf says: Квартальный отчёт"),
        "the pdf did not read back: {printed}"
    );
    assert!(
        printed.contains("docx says: Квартальный отчёт"),
        "the docx did not read back: {printed}"
    );
    assert!(
        std::fs::read(&pdf).unwrap().starts_with(b"%PDF-"),
        "not a pdf on disk"
    );
    assert!(
        std::fs::read(&docx).unwrap().starts_with(b"PK"),
        "not a docx on disk"
    );
}

/// The cell a report should cost: markdown in, a document out, no layout derived by hand.
#[tokio::test]
async fn a_report_is_markdown_handed_to_one_call() {
    let dir = workdir("report");
    let log = dir.join("session.jsonl");
    let pdf = dir.join("report.pdf");
    let code = format!(
        "import corpus_docs\n\
         from pypdf import PdfReader\n\
         report = '''# Квартальный отчёт\n\
         \n\
         Выручка **выросла** за квартал.\n\
         \n\
         - Север\n\
         - Юг\n\
         \n\
         | Регион | Доля |\n\
         |---|---|\n\
         | Север | 40% |\n\
         '''\n\
         corpus_docs.pdf(report, '{pdf}')\n\
         page = PdfReader('{pdf}').pages[0]\n\
         print('says:', ' '.join(page.extract_text().split()))\n\
         faces = page['/Resources']['/Font']\n\
         print('weights:', len({{str(faces[k]['/BaseFont']).split('+')[-1] for k in faces}}))\n",
        pdf = pdf.display(),
    );
    let endpoint = serve(vec![
        corpus_testkit::runs_python("call_1", &code),
        says("Written."),
    ])
    .await;

    corpus(
        &endpoint,
        &["Write the report.", "--log", log.to_str().unwrap()],
        &[],
    )
    .await;

    let printed = field(&log, "tool_stream", "text").concat();
    assert!(
        printed.contains("says: Квартальный отчёт Выручка выросла за квартал."),
        "the heading, the emphasis and the paragraph must survive: {printed}"
    );
    assert!(
        printed.contains("Регион Доля Север 40%"),
        "the table must survive as a table: {printed}"
    );
    // Regular, bold and the canvas default: emphasis that quietly rendered flat would
    // leave only two, and nothing else in the pipeline would say so.
    assert!(
        printed.contains("weights: 3"),
        "the bold weight was never embedded: {printed}"
    );
}

#[tokio::test]
async fn the_step_ceiling_can_be_raised_from_the_environment() {
    let dir = workdir("max-steps");
    let log = dir.join("session.jsonl");
    let endpoint = serve(vec![corpus_testkit::runs_python("call_1", "print('one')")]).await;

    corpus(
        &endpoint,
        &["Keep going.", "--log", log.to_str().unwrap()],
        &[("CORPUS_MAX_STEPS", "1")],
    )
    .await;

    assert_eq!(
        endpoint.requests().len(),
        1,
        "one step means one model call"
    );
    let ended = transcript(&log)
        .into_iter()
        .find(|event| event["t"] == "turn_end")
        .unwrap();
    assert_eq!(ended["stop"], "partial");
}

mod common;

use corpus_testkit::{kernel_dir, says, serve};
use serde_json::Value;

use common::{
    BIN, INTERRUPT_LINE, command, corpus, field, printed, run_line, served, transcript, workdir,
};

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
    use tokio::io::AsyncWriteExt;

    let endpoint = serve(vec![corpus_testkit::runs_python(
        "call_1",
        "print('working')\nimport time\ntime.sleep(30)",
    )])
    .await;
    let mut session = served(&endpoint);

    let turn = async {
        session.stdin.write_all(&run_line("go")).await.unwrap();
        let mut sent = false;
        while let Some(line) = session.lines.next_line().await.unwrap() {
            let event: Value = serde_json::from_str(&line).unwrap();
            if event["t"] == "tool_stream" && !sent {
                sent = true;
                session.stdin.write_all(INTERRUPT_LINE).await.unwrap();
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

/// The fanout the system prompt sells, over the whole path rather than a piece of it: the
/// model writes one cell, the cell makes one host call, and a model call per prompt runs
/// underneath it. The unit test beside `llm_batch` never leaves the host, so nothing there
/// proves the shim binds the name, that the answers survive the round trip into Python, or
/// that a cell whose host call outlasts its own timeout is left alone to finish.
#[tokio::test]
async fn a_cell_fans_a_batch_out_to_the_model() {
    let dir = workdir("batch");
    let log = dir.join("session.jsonl");
    let code = "answers = llm_batch(prompts=['one', 'two', 'three'])\n\
                print(len(answers), '|', '|'.join(sorted(answers)))\n";
    // Every prompt is answered the same, because the batch runs its prompts concurrently
    // and which one reaches the endpoint first is not something a test may depend on.
    let endpoint = serve(vec![
        corpus_testkit::runs_python("call_1", code),
        says("labelled"),
        says("labelled"),
        says("labelled"),
        says("Three labels."),
    ])
    .await;

    corpus(
        &endpoint,
        &["Label these.", "--log", log.to_str().unwrap()],
        &[],
    )
    .await;

    let printed = printed(&log);
    assert!(
        printed.contains("3 | labelled|labelled|labelled"),
        "the batch did not come back into the cell: {printed}"
    );
    assert_eq!(
        endpoint.requests().len(),
        5,
        "a step, one model call per prompt, and the step that answers"
    );
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
         import documents, docx\n\
         page = Canvas('{pdf}')\n\
         page.setFont(documents.unicode_font(), 14)\n\
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

    let printed = printed(&log);
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
        "import documents\n\
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
         documents.pdf(report, '{pdf}')\n\
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

    let printed = printed(&log);
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

/// The whole of the loader: the prompt names a file and the cell opens it. A path that came
/// out relative, or one the interpreter cannot reach, leaves every instruction the prompt
/// stopped carrying unreachable — and nothing else in a session would say so.
#[tokio::test]
async fn what_the_prompt_does_not_say_is_at_a_path_a_cell_can_open() {
    let dir = workdir("skills");
    let log = dir.join("session.jsonl");
    let skills = kernel_dir().join("skills").canonicalize().unwrap();
    let docs = skills.join("documents/SKILL.md");
    let endpoint = serve(vec![
        corpus_testkit::runs_python(
            "call_1",
            &format!("print(open('{}').read())", docs.display()),
        ),
        says("Read."),
    ])
    .await;

    corpus(
        &endpoint,
        &["Write a report.", "--log", log.to_str().unwrap()],
        &[],
    )
    .await;

    let prompt = &endpoint.requests()[0];
    for skill in ["shell", "documents"] {
        let manifest = skills.join(skill).join("SKILL.md");
        assert!(
            prompt.contains(&manifest.display().to_string()),
            "the prompt must name {skill} by the absolute path of its manifest: {prompt}"
        );
    }
    let printed = printed(&log);
    assert!(
        printed.contains("unicode_font"),
        "the cell opened the path the prompt gave it and the manifest did not come back: {printed}"
    );
}

/// One report's markdown becomes a deck, and the folder it sits in becomes searchable: both
/// skills through the kernel as it actually starts, because a package that imports in the
/// venv by hand proves nothing about the interpreter a session gets.
#[tokio::test]
async fn the_kernel_cuts_a_deck_and_searches_a_folder() {
    let dir = workdir("deck-and-search");
    let log = dir.join("session.jsonl");
    let deck = dir.join("q3.pptx");
    let code = format!(
        "import documents, search, slides\n\
         from pathlib import Path\n\
         report = '''# Отчёт\n\
         \n\
         ## Выручка\n\
         \n\
         - страховой резерв пересмотрен\n\
         \n\
         | Регион | Доля |\n\
         |---|---|\n\
         | Север | 40% |\n\
         '''\n\
         slides.deck(report, '{deck}')\n\
         from pptx import Presentation\n\
         show = Presentation('{deck}')\n\
         print('titles:', [s.shapes.title.text for s in show.slides])\n\
         print('body:', show.slides[1].placeholders[1].text_frame.text)\n\
         Path('{dir}/notes.md').write_text(report)\n\
         documents.pdf('# Договор\\n\\nСтраховой резерв формируется ежеквартально.', '{dir}/contract.pdf')\n\
         Path('{dir}/blank.txt').write_text('  ')\n\
         found = search.index('{dir}', stemmer='russian')\n\
         print('index:', found, 'empty:', [p.name for p in found.empty])\n\
         hit = found.find('страховой резерв')[0]\n\
         print('best:', Path(hit['path']).name, hit['page'], hit['snippet'][:40])\n",
        deck = deck.display(),
        dir = dir.display(),
    );
    let endpoint = serve(vec![
        corpus_testkit::runs_python("call_1", &code),
        says("The deck is cut and the folder is indexed."),
    ])
    .await;

    corpus(
        &endpoint,
        &["Cut a deck.", "--log", log.to_str().unwrap()],
        &[],
    )
    .await;

    let printed = printed(&log);
    assert!(
        printed.contains("titles: ['Отчёт', 'Выручка', 'Выручка']"),
        "a heading opens a slide and the table takes one of its own: {printed}"
    );
    assert!(
        printed.contains("body: страховой резерв пересмотрен"),
        "the bullet did not reach the slide: {printed}"
    );
    assert!(
        std::fs::read(&deck).unwrap().starts_with(b"PK"),
        "not a pptx on disk"
    );
    assert!(
        printed.contains("empty: ['blank.txt']"),
        "a file that yielded no text must be named rather than counted as indexed: {printed}"
    );
    assert!(
        printed.contains("best: contract.pdf 1 "),
        "the phrase is in the pdf and the pdf is where the search must land: {printed}"
    );
}

/// The loop `skill-creator` describes, end to end: a session writes a skill into a root that
/// did not exist when it started, uses it in the next cell, and the session after that is
/// told about it in its prompt. The cache drop is the part that fails silently — without it
/// the import raises as though the skill were wrong rather than one call late.
#[tokio::test]
async fn a_session_writes_a_skill_and_uses_it_without_starting_again() {
    let dir = workdir("authoring");
    let home = dir.join("home");
    std::fs::create_dir_all(&home).unwrap();
    let log = dir.join("session.jsonl");
    let write = "\
from pathlib import Path\n\
skill = Path('.corpus/skills/tariffs')\n\
skill.mkdir(parents=True)\n\
(skill / 'SKILL.md').write_text('---\\nname: tariffs\\ndescription: what this port charges\\n---\\n')\n\
(skill / '__init__.py').write_text('def berth(tons):\\n    return tons * 3\\n')\n\
print('written:', skill)\n";
    let use_it = "\
import importlib\n\
importlib.invalidate_caches()\n\
import tariffs\n\
print('berth:', tariffs.berth(4))\n";
    let endpoint = serve(vec![
        corpus_testkit::runs_python("call_1", write),
        corpus_testkit::runs_python("call_2", use_it),
        says("Written and used."),
        says("Ready."),
    ])
    .await;

    let mut first = command(&endpoint);
    first.current_dir(&dir).env("HOME", &home).args([
        "Write a skill.",
        "--log",
        log.to_str().unwrap(),
    ]);
    let output = first.output().await.unwrap();
    assert!(
        output.status.success(),
        "corpus failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let printed = printed(&log);
    assert!(
        printed.contains("berth: 12"),
        "the skill written a cell ago must import and run in the same session: {printed}"
    );

    // And the session after it is the one that learns about it from the prompt.
    let mut next = command(&endpoint);
    next.current_dir(&dir)
        .env("HOME", &home)
        .arg("Anything else?");
    assert!(next.output().await.unwrap().status.success());
    // The endpoint holds both runs' requests, and the second run's only one is the last.
    let prompt = endpoint.requests().pop().unwrap();
    assert!(
        prompt.contains("tariffs — what this port charges"),
        "the next session's prompt must offer the skill: {prompt}"
    );
}

/// A skill a project keeps for itself: named in the prompt by the path a cell can open,
/// importable in that cell, and ahead of the one the checkout ships under the same name —
/// which is the whole reason for looking anywhere but beside the kernel.
#[tokio::test]
async fn a_project_keeps_skills_of_its_own_and_they_come_first() {
    let dir = workdir("project-skills");
    let log = dir.join("session.jsonl");
    let own = dir.join(".corpus/skills");
    for (skill, description, module) in [
        (
            "inventory",
            "counts what the warehouse holds",
            "def count():\n    return 7\n",
        ),
        (
            "shell",
            "the project's own commands",
            "def sh(command):\n    return 0\n",
        ),
    ] {
        std::fs::create_dir_all(own.join(skill)).unwrap();
        std::fs::write(
            own.join(skill).join("SKILL.md"),
            format!("---\nname: {skill}\ndescription: {description}\n---\n"),
        )
        .unwrap();
        std::fs::write(own.join(skill).join("__init__.py"), module).unwrap();
    }
    // A home of its own, so whatever the person running the tests keeps in theirs is not
    // what this asserts about.
    let home = dir.join("home");
    std::fs::create_dir_all(&home).unwrap();

    let endpoint = serve(vec![
        corpus_testkit::runs_python("call_1", "import inventory\nprint(inventory.count())"),
        says("Counted."),
    ])
    .await;
    let output = command(&endpoint)
        .current_dir(&dir)
        .env("HOME", &home)
        .args(["Count the stock.", "--log", log.to_str().unwrap()])
        .output()
        .await
        .unwrap();
    assert!(
        output.status.success(),
        "corpus failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let prompt = &endpoint.requests()[0];
    let own = own.canonicalize().unwrap();
    for skill in ["inventory", "shell"] {
        let manifest = own.join(skill).join("SKILL.md");
        assert!(
            prompt.contains(&manifest.display().to_string()),
            "the prompt must name the project's {skill} at its own path: {prompt}"
        );
    }
    let shipped = kernel_dir()
        .join("skills/shell/SKILL.md")
        .canonicalize()
        .unwrap();
    assert!(
        !prompt.contains(&shipped.display().to_string()),
        "the project's version of the skill must be the only one offered: {prompt}"
    );
    assert!(
        printed(&log).contains('7'),
        "the cell must import the project's skill: {}",
        printed(&log)
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

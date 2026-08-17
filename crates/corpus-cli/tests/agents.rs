//! Agents a session sends off, over the whole path: the cell that spawns one, the child's
//! own loop, and the answer coming back into the cell that asked for it.

mod common;

use corpus_testkit::{runs_python, says, serve};
use serde_json::Value;

use common::{BIN, Step, ask, command, ends_turn, events, printed, served, succeeds, workdir};

/// Ctrl-C stops the cell a person is watching, and nothing else. The agents that cell
/// sent off keep working — killing them would throw away the very thing being waited on —
/// and the handles are still in the namespace on the other side of the interrupt.
#[tokio::test]
async fn stopping_the_cell_that_waits_does_not_stop_what_it_waits_for() {
    let waiting = "kid = spawn('a job that takes a while')\n\
                   print('waiting')\n\
                   print('answered:', kid.result(timeout=300))\n";
    let looking = "print('still mine:', [agent.task for agent in agents()], 'done:', kid.done())";
    let endpoint = serve(vec![
        runs_python("call_1", waiting),
        // The child's own cell, which keeps it busy for longer than this test lives.
        runs_python("call_2", "import time; time.sleep(30)"),
        runs_python("call_3", looking),
        says("It is still at it."),
    ])
    .await;

    let mut served = served(&endpoint);

    let mut stopped = false;
    served
        .turn("Send one off and wait.", |event| {
            // The cell says when it has reached the wait, which is the only moment an
            // interrupt is testing anything.
            let waiting = event["t"] == "tool_stream"
                && event["text"]
                    .as_str()
                    .is_some_and(|text| text.contains("waiting"));
            match !stopped && waiting {
                true => {
                    stopped = true;
                    Step::Interrupt
                }
                false => ends_turn(event),
            }
        })
        .await;

    let printed: String = served
        .turn("What have you got?", ends_turn)
        .await
        .iter()
        .filter(|event| event["t"] == "tool_stream")
        .filter_map(|event| event["text"].as_str())
        .collect();
    assert!(
        printed.contains("still mine: ['a job that takes a while']"),
        "the handle must survive the interrupt: {printed}"
    );
    assert!(
        printed.contains("done: False"),
        "the agent must still be working: {printed}"
    );
}

/// Every agent's stream shares the one pipe `connect` reads, so the turn this side is
/// waiting on has to be named. Taking the first ending that goes past would hand a turn
/// back while it is still running, and whatever is typed next lands on top of it.
#[tokio::test]
async fn a_child_finishing_does_not_end_the_turn_a_client_is_reading() {
    let dir = workdir("connect-child");
    let log = dir.join("client.jsonl");
    let endpoint = serve(vec![
        runs_python(
            "call_1",
            "kid = spawn('go and look')\nwhile kid.result(timeout=10) is None: pass\n",
        ),
        says("What the agent found."),
        says("Both are in."),
    ])
    .await;

    succeeds(command(&endpoint).args([
        "connect",
        "Send one off.",
        "--log",
        log.to_str().unwrap(),
        "--",
        BIN,
        "serve",
    ]))
    .await;

    let events = events(&log);
    // The child's turn and the session's own, and possibly the turn the news then
    // started over there: what matters is that more than one ending came past.
    assert!(
        events.iter().filter(|e| e["t"] == "turn_end").count() >= 2,
        "every agent's stream shares this pipe, so more than one ending crosses it"
    );
    assert!(
        events
            .iter()
            .any(|event| event["t"] == "answer" && event["text"] == "Both are in."),
        "the turn must be read to its own end, not to the first end that goes past"
    );
}

/// An answer that cost tokens, so a session can be billed for what its agents spend.
fn says_costing(text: &str, input: u32, output: u32) -> Vec<u8> {
    corpus_testkit::sse(&[
        &corpus_testkit::content(text),
        &format!(
            r#"{{"choices":[{{"index":0,"delta":{{}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":{input},"completion_tokens":{output}}}}}"#
        ),
    ])
}

/// What the children spend is spent by the session, so it lands on the turn's bill. It
/// does not land on the window readout: a child fills a window of its own, and a session
/// that read 100k tokens through its agents has not filled its own context by 100k.
#[tokio::test]
async fn what_an_agent_spends_is_on_its_parents_bill_and_not_in_its_context() {
    let dir = workdir("usage");
    let log = dir.join("session.jsonl");
    let endpoint = serve(vec![
        // The cell waits, so the child is finished and counted before the turn ends.
        runs_python(
            "call_1",
            "kid = spawn('read it')\nwhile kid.result(timeout=10) is None: pass\n",
        ),
        says_costing("What the agent found.", 500, 200),
        says_costing("Read.", 11, 2),
    ])
    .await;

    ask(&endpoint, "Read it through an agent.", &log).await;

    let ended = events(&log)
        .into_iter()
        .rfind(|event| event["t"] == "turn_end")
        .expect("the turn ended");
    assert_eq!(
        ended["usage"]["input"], 511,
        "the agent's 500 belong on the bill beside the session's own 11: {ended}"
    );
    assert_eq!(ended["usage"]["output"], 202, "{ended}");
    assert_eq!(
        ended["context"], 11,
        "how full the window is was never the sum of every agent's: {ended}"
    );
}

/// A child coming back rings the doorbell: the session takes a turn nobody typed, and is
/// told which agent finished and roughly what it holds — not the answer itself, which is
/// one `result()` away and would otherwise sit in the context for good.
#[tokio::test]
async fn an_agent_coming_back_is_a_turn_the_session_takes_on_its_own() {
    let long = "The numbers are 3, 5 and 8, and here is a great deal more about them: ".repeat(6);
    let endpoint = serve(vec![
        runs_python(
            "call_1",
            "kid = spawn('dig up the numbers')\nwhile kid.result(timeout=10) is None: pass\n",
        ),
        says(&long),
        says("One is off digging."),
        says("Right, it came back with the numbers."),
    ])
    .await;

    let mut told = false;
    let seen = served(&endpoint)
        .turn("Find the numbers.", |event| {
            // Every agent's turns are in this stream, the child's among them, so the end
            // being waited for is the end of the turn the news itself started.
            let done = told && event["t"] == "turn_end";
            told |= event["t"] == "notice";
            match done {
                true => Step::Stop,
                false => Step::Go,
            }
        })
        .await;

    let notice = seen
        .iter()
        .find(|event| event["t"] == "notice")
        .expect("news from the children is its own kind of event, not a forged question");
    let told = notice["text"].as_str().unwrap();
    assert!(
        told.contains("finished") && told.contains("The numbers are 3, 5 and 8"),
        "the doorbell says who finished and enough to place it: {told}"
    );
    assert!(
        told.contains("UNTRUSTED CONTENT"),
        "what is quoted in it is an agent's own words, and an agent quotes what it read: {told}"
    );
    assert!(
        told.chars().count() < long.chars().count(),
        "a doorbell is not the delivery: {told}"
    );
    assert_eq!(
        seen.iter()
            .filter(|event| event["t"] == "user_message" && event["agent"] == notice["agent"])
            .count(),
        1,
        "the session was asked one thing by a person, and that is the only question in it"
    );
    let woken = endpoint.requests().last().unwrap().clone();
    assert!(
        woken.contains("not the person you are talking to"),
        "the model must be able to tell who is talking: {woken}"
    );
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

    ask(&endpoint, "Read the second report.", &log).await;

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
        log.iter().any(|event| event["t"] == "answer"
            && event["agent"] == started["agent"]
            && event["text"] == "Margins fell by a fifth."),
        "the child's own stream belongs in the log under its own name"
    );
}

/// More work for an agent that already has some. The inbox is the queue, so `send` never
/// refuses and never waits; what comes back once it has settled is its latest turn, and
/// the turn after remembers the turn before, because it is one agent throughout.
#[tokio::test]
async fn more_work_for_a_busy_agent_waits_in_its_inbox() {
    let dir = workdir("send");
    let log = dir.join("session.jsonl");
    let code = "kid = spawn('count the first report')\n\
                kid.send('now the second, in the same words')\n\
                while (answer := kid.result(timeout=10)) is None:\n\
                \x20   pass\n\
                print('last of it:', answer)\n\
                print('the ones I have:', [agent.task for agent in agents()])\n";
    let endpoint = serve(vec![
        runs_python("call_1", code),
        says("The first has nine tables."),
        says("The second has four."),
        says("Nine and four."),
    ])
    .await;

    ask(&endpoint, "Count the tables in both.", &log).await;

    let printed = printed(&log);
    assert!(
        printed.contains("last of it:") && printed.contains("The second has four."),
        "what settles last is what result gives back: {printed}"
    );
    assert!(
        printed.contains("the ones I have: ['count the first report']"),
        "a handle outlives the turn it was made in: {printed}"
    );
    let second = &endpoint.requests()[2];
    assert!(
        second.contains("The first has nine tables.") && second.contains("now the second"),
        "the second turn is the same agent's, and it remembers the first: {second}"
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

    ask(&endpoint, "What can the agent you send do?", &log).await;

    let printed = printed(&log);
    assert!(
        printed.contains("mine: ['fetch_url', 'llm_batch']"),
        "a child's namespace must hold the tools and not the delegation: {printed}"
    );
}

/// Delivery over the whole path: the name is in the cell's namespace, the host checks the
/// file, and the session's log carries what a consumer needs to go and read it.
#[tokio::test]
async fn a_file_a_cell_produced_leaves_the_session_as_an_event() {
    let dir = workdir("delivery");
    let log = dir.join("session.jsonl");
    let writing = "from pathlib import Path\n\
                   Path('report.md').write_text('# what I found\\n')\n\
                   print(send_user_file(path='report.md', caption='what I found'))\n\
                   print('outside:', end=' ')\n\
                   try:\n\
                   \x20   send_user_file(path='/etc/hosts')\n\
                   except HostError as refusal:\n\
                   \x20   print(refusal)\n";
    let endpoint = serve(vec![
        runs_python("call_1", writing),
        says("Sent the report."),
    ])
    .await;

    succeeds(command(&endpoint).current_dir(&dir).args([
        "Write the report and send it.",
        "--log",
        log.to_str().unwrap(),
    ]))
    .await;

    let printed = printed(&log);
    assert!(
        printed.contains("outside: refused") && printed.contains("outside the working directory"),
        "a path out of the working directory is refused in the cell: {printed}"
    );
    let sent: Vec<Value> = events(&log)
        .into_iter()
        .filter(|event| event["t"] == "user_file")
        .collect();
    assert_eq!(sent.len(), 1, "one call, one delivery: {sent:?}");
    assert_eq!(sent[0]["filename"], "report.md");
    assert_eq!(sent[0]["bytes"], 15);
    assert_eq!(sent[0]["caption"], "what I found");
    assert_eq!(
        sent[0]["path"],
        dir.join("report.md")
            .canonicalize()
            .unwrap()
            .to_str()
            .unwrap(),
        "the consumer opens this path itself"
    );
}

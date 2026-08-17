# Corpus Harness

An agent whose loop runs where your data is.

Corpus gives a model one tool — a Python interpreter that keeps its variables between
calls — and lets it work on your material there instead of pulling it all into a context
window. Search, fetching, batching, delegation and file delivery are functions already
bound in that namespace, not tools the model has to be handed one at a time.

Hosted deployment: <https://corpus.alphacompute.dev/> (SSO, work email).

## The idea

Corpus is built on the RLM paradigm — Recursive Language Models,
[arXiv:2512.24601](https://arxiv.org/abs/2512.24601). The paper's claim is that a long
prompt should be treated as an _environment_ rather than as context: the model examines
it, decomposes it, and calls itself over the pieces, which lets it work over inputs far
larger than the window it was trained with.

Corpus is that idea as a working harness:

| RLM concept                  | In Corpus                                                                                     |
| ---------------------------- | --------------------------------------------------------------------------------------------- |
| The prompt as an environment | A persistent Python namespace; data lives in variables, and only slices of it reach the model |
| Recursive self-call          | `llm_batch(prompts=[...])` — one model call per prompt, all inside one cell                   |
| Sub-agent recursion          | `spawn(task=...)` — a full agent with its own interpreter and its own turn                    |
| Depth control                | A spawned agent is a leaf: `spawn` is simply not bound in its namespace                       |

So there are three ways to spend a model call, and choosing between them is most of the
work: a question that is one call is one call, the same small question asked of two
hundred chunks is `llm_batch`, and an errand — go read this, dig through that, come back
with what you found — is an agent.

## Quick start

Prerequisites: a Rust toolchain and `python3` on PATH.

```sh
git clone <this repo> && cd corpus-harness
cargo build --release
```

Point it at a provider. Any OpenAI-compatible endpoint works — Corpus only speaks
`GET /models` and `POST /chat/completions`. The examples here use the `dev.shroud.us`
gateway:

```sh
cat > .env <<'EOF'
CORPUS_BASE_URL=https://dev-gateway.shroud.us/v1
CORPUS_API_KEY=sk-your-key
CORPUS_MODEL=nemotron-3-ultra
EOF
```

`.env` is read from the working directory at startup and never overrides what the shell
already set. Leave `CORPUS_MODEL` out and Corpus takes the first model the provider
lists.

Then run it:

```sh
./target/release/corpus                       # a conversation, in the terminal UI
./target/release/corpus "how many rows in sales.csv, and what is the median order?"
```

To install the binary, keep the checkout's `kernel/` reachable — that directory holds the
interpreter shim and the shipped skills:

```sh
cargo install --path crates/corpus-cli
export CORPUS_KERNEL_DIR=/path/to/corpus-harness/kernel
```

On first run Corpus builds a virtualenv at `kernel/.venv` holding whatever the installed
skills ask for. That takes a minute once; afterwards it is rebuilt only when the list of
requirements changes.

## How a turn works

1. You ask something. The model gets one tool: `python`, taking `{"code": "..."}`.
2. It writes a cell. The cell runs in a Python process that outlives it, so variables,
   imports and `_` (the value of the last cell that ended on an expression) persist.
3. Printed output streams back as it happens; the value the cell ended on comes back as a
   repr. Both are truncated into the model's context at 8 000 characters — but the whole
   value is still alive in the interpreter, one slice away.
4. The model reads what came back and writes the next cell against it.
5. When it stops calling the tool, its text is the answer. Text written _beside_ a tool
   call is never treated as an answer.

A turn ends on its own budget: 200 steps, 15 minutes wall clock, 120 seconds per cell.
When the transcript grows past 120 000 characters, the oldest tool output is dropped
first — a page can be fetched again, a line of reasoning cannot.

## What is bound in the namespace

Every session gets these:

```python
web_search(query=..., count=10)   # -> [{title, url, snippet}]
fetch_url(url=...)                # -> {url, status, text}
llm_batch(prompts=[...])          # -> list[str], one answer per prompt, in order
```

A session talking to a person also gets:

```python
send_user_file(path=..., caption=None)   # hand a file over; only paths under the cwd
spawn(task=...)                          # -> an agent handle, already working
agents()                                 # -> the agents you have
```

An agent handle carries `kid.result(timeout=30)` (what its last finished turn answered,
or `None` while it is still working), `kid.send(text)` and `kid.done()`. Nothing blocks
forever — `result` waits at most 300 seconds — so a row of agents is collected in a loop
rather than waited on one at a time.

None of these run inside the interpreter. The cell sends a request over a pipe, the Rust
host performs it and sends the value back. That is what keeps credentials out of
model-written code.

A worked example of the three levers, as the model would write it:

```python
rows = open("support-tickets.csv").read().splitlines()
len(rows)                                        # peek: 41 812 lines, not in context
chunks = ["\n".join(rows[i:i+50]) for i in range(1, len(rows), 50)]
labels = llm_batch([f"Label the sentiment of each line.\n{c}" for c in chunks])
sum(1 for l in labels if l.startswith("ERROR:"))  # count failures before trusting it
```

Eight prompts are in flight at a time and the whole batch has 300 seconds; a prompt that
failed comes back as a string opening with `ERROR:` in its own place, so alignment with
the input list always holds.

## Skills

A skill is a directory holding a `SKILL.md` and, usually, the Python package it
describes. The prompt carries each skill's name, one-line description and path; the model
opens the file itself, in the session where it needs it. Skills roots, most specific
first:

| Root               | Whose it is             |
| ------------------ | ----------------------- |
| `./.corpus/skills` | This project's          |
| `~/.corpus/skills` | This person's           |
| `kernel/skills`    | What the checkout ships |

A name found twice is the first one, so a project can ship its own version of a shipped
skill. Each root is on the interpreter's path, so the directory name is what `import`
takes.

The checkout ships: `charts`, `citations`, `documents`, `queries`, `scans`, `search`,
`shell`, `skill-creator`, `slides`, `spreadsheets`.

A `SKILL.md` needs frontmatter with a `description` — a skill without one is invisible in
the prompt and is skipped with a warning. Anything a skill needs installed goes in a
`requirements.txt` beside it.

## Configuration

| Variable                   | Meaning                                                                                             |
| -------------------------- | --------------------------------------------------------------------------------------------------- |
| `CORPUS_BASE_URL`          | OpenAI-compatible endpoint. Default `https://api.openai.com/v1`                                     |
| `CORPUS_API_KEY`           | Provider key. `OPENAI_API_KEY` is read as a fallback                                                |
| `CORPUS_MODEL`             | Model id. Unset means the first one the provider lists                                              |
| `CORPUS_CONTEXT_WINDOW`    | Tokens the model takes at once, for the on-screen readout, when the provider's listing does not say |
| `CORPUS_SEARCH_URL`        | Search endpoint. Defaults to Brave's when only a key is given                                       |
| `CORPUS_SEARCH_KEY`        | Search key. Without either variable the session has no `web_search` and says so                     |
| `CORPUS_PYTHON`            | Interpreter to run cells with, instead of the managed virtualenv                                    |
| `CORPUS_KERNEL_DIR`        | Where `corpus_kernel/` and the shipped `skills/` live                                               |
| `CORPUS_LOG`               | Directory to write session logs into                                                                |
| `CORPUS_TRACE`             | File to record the raw provider wire into, for reporting a bad stream                               |
| `CORPUS_MAX_STEPS`         | Steps per turn. Default 200                                                                         |
| `CORPUS_WALL_CLOCK_SECS`   | Seconds per turn. Default 900                                                                       |
| `CORPUS_CELL_TIMEOUT_SECS` | Seconds per cell. Default 120                                                                       |

## Running it

```sh
corpus                            # terminal UI: Ctrl-C stops a running turn, or leaves when idle
corpus "one question"             # one answer on stdout, then exit
corpus --log run.jsonl "..."      # record the session
corpus --resume run.jsonl         # continue it — dialogue only; the interpreter is fresh
corpus serve                      # speak the session protocol on stdin/stdout
corpus connect -- ssh box corpus serve
```

A prompt on the command line or a pipe on stdout means something is reading the output,
so those paths stay plain text; the UI is only for a person at a terminal.

`serve` reads one JSON command per line and writes one JSON event per line:

```json
{"cmd":"run","text":"..."}
{"cmd":"interrupt"}
{"cmd":"exit"}
```

Events are the session log format — `session_start`, `turn_start`, `user_message`,
`code_delta`, `tool_start`, `tool_stream`, `tool_end`, `message_delta`, `agent_start`,
`agent_end`, `user_file`, `answer`, `turn_end` — each tagged with `t` and with the agent
it belongs to. The log _is_ the session: the model's context is derived from it rather
than stored beside it, which is what lets compaction drop tool output without losing the
thread.

## Boundaries

Corpus runs model-written code, so the edges are drawn deliberately.

- **The kernel inherits no secrets.** The interpreter is spawned with the environment
  cleared, keeping only `PATH`, `HOME`, `LANG` and `TMPDIR`. Provider and search keys
  stay on the Rust side of the pipe and no cell ever sees them.
- **Fetching cannot reach inward.** A URL is resolved once, refused if it lands on a
  private, loopback, link-local or carrier-grade-NAT address, and then connected to that
  exact address, so nothing can change between the check and the request. Only `http` and
  `https`; redirects are not followed. When the host is behind a proxy, the check is left
  to the proxy that is doing the dialling.
- **Fetched text is fenced.** Pages, search snippets and agents' answers arrive wrapped in
  an `UNTRUSTED CONTENT` marker that names its source and says, in the text itself, that
  this is material to report on and never instructions to follow.
- **Answers are scrubbed.** Anything shaped like a credential is replaced with a visible
  `[secret removed]` before the text leaves the agent.
- **Delivery is confined.** `send_user_file` takes only regular files under the working
  directory, resolved canonically so a symlink out of the tree is refused by where it
  lands rather than allowed by how it was spelled, and at most 64 MB.
- **Recursion is one level deep.** A spawned agent has no `spawn` bound and its prompt
  never mentions one. Eight children may take a turn at once; the rest queue.

The interpreter itself is not a sandbox — a cell can read the filesystem and run
subprocesses. Run Corpus where you would run a shell with the same reach.

## Layout

| Crate                    | What it is                                                              |
| ------------------------ | ----------------------------------------------------------------------- |
| `crates/corpus-agent`    | The turn loop: messages, tool calls, compaction, budgets, the event log |
| `crates/corpus-kernel`   | The Python subprocess and its JSON-lines protocol                       |
| `crates/corpus-provider` | The OpenAI-compatible client: streaming, model listing, wire tracing    |
| `crates/corpus-cli`      | Host functions, children, skills, the terminal UI, the session protocol |
| `crates/corpus-testkit`  | A fake provider for the tests                                           |
| `kernel/`                | The interpreter shim (`corpus_kernel/`) and the shipped `skills/`       |

```sh
cargo test
```

"""JSON-lines kernel: cells run on the main thread, the host talks on a reader thread."""

import ast
import json
import os
import queue
import signal
import sys
import threading
import traceback

MAX_REPR = 4000

# What the namespace summary may say about one cell. A run that binds two hundred names
# is a run whose interesting ones are the few it just touched, and a panel nobody can read
# is worth less than a short one that is true.
MAX_NAMES = 40
MAX_NAME_REPR = 60

# Values small enough that showing them says more than their type does. Anything else is
# described rather than printed: `repr` on a two-hundred-thousand-row frame builds the
# whole string before anyone can truncate it, and this runs after every cell.
_SHOWN = (bool, int, float, complex, str, bytes)

# The protocol must survive `print` inside a cell forging a frame, so it moves off fd 1
# before anything else runs: fd 1 becomes stderr (subprocesses can only pollute the log),
# sys.stdout becomes a proxy that emits `stream` frames.
# Text a cell chunked out of a mixed-encoding file can hold a lone surrogate, which no
# encoder will take: it is replaced rather than raised, because the frame is the only way
# the cell has of saying anything at all, including that something went wrong.
_proto = os.fdopen(
    os.dup(1), "w", buffering=1, encoding="utf-8", errors="replace", newline="\n"
)
os.dup2(2, 1)

_proto_lock = threading.Lock()
_current_id = ""


def _send(**frame):
    line = json.dumps(frame, ensure_ascii=False)
    with _proto_lock:
        _proto.write(line + "\n")
        _proto.flush()


class _StreamProxy:
    encoding = "utf-8"

    def write(self, text):
        if text:
            _send(type="stream", id=_current_id, text=text)
        return len(text)

    def flush(self):
        pass

    def isatty(self):
        return False

    def writable(self):
        return True


class HostError(Exception):
    pass


_pending = {}
_pending_lock = threading.Lock()


def _host_fn(name):
    counter = [0]

    def call(**kwargs):
        # The shim mints no identifiers of its own: req_id is derived from the cell id the
        # host assigned, so it stays unique without a uuid dependency. `cell` goes as its
        # own field rather than leaving the host to read it back out of req_id, which would
        # put the spelling of one identifier in two languages. The cell is what the main
        # thread is running, and only calls from there carry it: a thread the cell spawned
        # outlives the cell, and its calls must not pass for the cell's own progress.
        counter[0] += 1
        cell = _current_id if threading.current_thread() is threading.main_thread() else ""
        req_id = f"{cell}#{name}{counter[0]}"
        event = threading.Event()
        with _pending_lock:
            _pending[req_id] = [event, None]
        _send(type="host_request", req_id=req_id, cell=cell, fn=name, args=kwargs)
        try:
            event.wait()
        finally:
            # The wait is where an interrupt lands, and interrupting a cell mid-call is an
            # ordinary gesture rather than a rarity: a slot left behind on every one of them
            # is a leak that grows for as long as the kernel lives.
            with _pending_lock:
                reply = _pending.pop(req_id)[1]
        if not reply.get("ok"):
            raise HostError(reply.get("error", "host call failed"))
        return reply.get("value")

    call.__name__ = name
    return call


def _reader():
    # Ends when the host closes the pipe, however the host ended: a kernel nobody can
    # talk to would otherwise sit here holding memory until the machine is rebooted.
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        kind = msg.get("type")
        if kind == "exec":
            _cells.put(msg)
        elif kind == "interrupt":
            # A real signal, not _thread.interrupt_main(): only a signal wakes a main
            # thread blocked in a syscall, which is where a cell worth interrupting sits.
            os.kill(os.getpid(), signal.SIGINT)
        elif kind == "host_reply":
            with _pending_lock:
                slot = _pending.get(msg.get("req_id"))
                if slot is not None:
                    slot[1] = msg
                    slot[0].set()
    os._exit(0)


_cells = queue.Queue()


def _safe_repr(value):
    try:
        text = repr(value)
    except BaseException as exc:
        text = f"<unrepresentable: {exc!r}>"
    if len(text) <= MAX_REPR:
        return text
    # Only ever the tail value, which `_run` has just bound to `_`, so the pointer is honest:
    # the reader can go and slice the thing instead of mourning it.
    return f"{text[:MAX_REPR]}... [truncated; the full value is in `_` (repr: {len(text)} chars)]"


def _describe(value):
    """A binding as a reader needs it: what it is, how big, and its value only when the
    value is small enough to be the more useful of the two."""
    kind = type(value).__name__
    size = None
    shown = None
    try:
        shape = getattr(value, "shape", None)
        if isinstance(shape, tuple) and shape:
            size = "\u00d7".join(str(part) for part in shape)
        elif not isinstance(value, (str, bytes)):
            size = str(len(value))
    except BaseException:
        size = None
    if isinstance(value, _SHOWN):
        try:
            text = repr(value)
        except BaseException:
            text = None
        if text is not None and len(text) <= MAX_NAME_REPR:
            shown = text
        elif isinstance(value, (str, bytes)):
            size = f"{len(value)} chars" if isinstance(value, str) else f"{len(value)} bytes"
    return {"name": None, "type": kind, "size": size, "repr": shown}


def _namespace(ns, floor, seen):
    """What this cell left behind that the one before it did not.

    Only the difference: a cell that touched one name should read as having touched one
    name. `floor` is what the harness bound before any cell ran, which is furniture rather
    than the model's work — until the model rebinds one, and then it is worth saying.
    """
    now = {}
    for name, value in list(ns.items()):
        if name.startswith("__") or name == "_":
            continue
        if name in floor and ns[name] is floor[name]:
            continue
        now[name] = _describe(value)

    changed = []
    for name, about in now.items():
        if seen.get(name) != about:
            about = dict(about, name=name)
            changed.append(about)
    gone = [name for name in seen if name not in now]

    seen.clear()
    seen.update(now)

    # One budget for both: a cell that deletes two hundred tracked names would
    # otherwise send every one of them, and a frame nobody can read is worth less
    # than a short one that says how much it left out.
    trimmed = max(0, len(changed) - MAX_NAMES)
    changed = changed[:MAX_NAMES]
    room = MAX_NAMES - len(changed)
    trimmed += max(0, len(gone) - room)
    gone = gone[:room]
    return changed, gone, trimmed


def _left(ns, floor, seen):
    """A cell that raised bound whatever it got through before it did."""
    if seen is None:
        return {}
    names, gone, trimmed = _namespace(ns, floor, seen)
    return {"names": names, "gone": gone, "trimmed": trimmed}


def _run(cell_id, code, ns, floor=None, seen=None):
    global _current_id
    _current_id = cell_id
    try:
        block = ast.parse(code, "<cell>", "exec")
        tail = None
        if block.body and isinstance(block.body[-1], ast.Expr):
            tail = ast.Expression(block.body.pop().value)
            ast.fix_missing_locations(tail)
        exec(compile(block, "<cell>", "exec"), ns)
        value = eval(compile(tail, "<cell>", "eval"), ns) if tail is not None else None
        if value is not None:
            # IPython's bargain, for IPython's reason: a result too big to read is not lost,
            # it is one name away. None is skipped so a statement cell cannot erase it.
            ns["_"] = value
        names, gone, trimmed = (
            _namespace(ns, floor, seen) if seen is not None else ([], [], 0)
        )
        _send(
            type="done",
            id=cell_id,
            status="ok",
            repr="" if value is None else _safe_repr(value),
            names=names,
            gone=gone,
            trimmed=trimmed,
        )
    except KeyboardInterrupt:
        _send(type="done", id=cell_id, status="error",
              traceback="KeyboardInterrupt: cell interrupted", **_left(ns, floor, seen))
    except BaseException:
        _send(type="done", id=cell_id, status="error",
              traceback=traceback.format_exc(), **_left(ns, floor, seen))
    finally:
        _current_id = ""


def _bind(ns, fns):
    """Binds the host's functions, and the one object the cell would otherwise have to
    build for itself: an agent read back from an id is three calls that all take the same
    id, and the model writing that by hand is the wrapper we would rather it did not.

    Depth lives in `fns` and nowhere else: with no `spawn` in the list there is no agent
    class, no `agents()`, and nothing in the namespace to suggest either exists.
    """
    raw = {name: _host_fn(name) for name in fns}
    ns.update({name: call for name, call in raw.items() if name not in ("result", "send", "done")})
    if "spawn" not in raw:
        return

    class Agent:
        __slots__ = ("id", "task")

        def __init__(self, id, task=""):
            self.id = id
            self.task = task

        def result(self, timeout=30):
            """What its last finished turn answered, or None while it is still working."""
            return raw["result"](agent=self.id, timeout=timeout)

        def send(self, text):
            """Another turn for it; it takes this one when the one it is on is done."""
            return raw["send"](agent=self.id, text=text)

        def done(self):
            return raw["done"](agent=self.id)

        def __repr__(self):
            return f"<agent {self.id}: {self.task}>"

    ns["spawn"] = lambda task: Agent(raw["spawn"](task=task), task)
    ns["agents"] = lambda: [Agent(kid["agent"], kid["task"]) for kid in raw["agents"]()]


def main():
    init = json.loads(sys.stdin.readline())
    # `_` is bound from the start, so a peek written before anything returned a value
    # reads as empty rather than raising over a name the prompt promised.
    ns = {"__name__": "__corpus__", "HostError": HostError, "_": ""}
    _bind(ns, init.get("fns", []))
    # Taken by identity, so a name the model rebinds stops being furniture and starts
    # being its work — which is exactly when it becomes worth reporting.
    floor = dict(ns)
    seen = {}

    sys.stdout = _StreamProxy()
    sys.stderr = _StreamProxy()
    threading.Thread(target=_reader, daemon=True).start()
    _send(type="ready")

    while True:
        try:
            cell = _cells.get()
        except KeyboardInterrupt:
            continue  # interrupt landed between cells
        _run(cell["id"], cell["code"], ns, floor, seen)


if __name__ == "__main__":
    main()

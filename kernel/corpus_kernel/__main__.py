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
        event.wait()
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


def _run(cell_id, code, ns):
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
        _send(
            type="done",
            id=cell_id,
            status="ok",
            repr="" if value is None else _safe_repr(value),
        )
    except KeyboardInterrupt:
        _send(type="done", id=cell_id, status="error", traceback="KeyboardInterrupt: cell interrupted")
    except BaseException:
        _send(type="done", id=cell_id, status="error", traceback=traceback.format_exc())
    finally:
        _current_id = ""


def main():
    init = json.loads(sys.stdin.readline())
    # `_` is bound from the start, so a peek written before anything returned a value
    # reads as empty rather than raising over a name the prompt promised.
    ns = {"__name__": "__corpus__", "HostError": HostError, "_": ""}
    for name in init.get("fns", []):
        ns[name] = _host_fn(name)

    sys.stdout = _StreamProxy()
    sys.stderr = _StreamProxy()
    threading.Thread(target=_reader, daemon=True).start()
    _send(type="ready")

    while True:
        try:
            cell = _cells.get()
        except KeyboardInterrupt:
            continue  # interrupt landed between cells
        _run(cell["id"], cell["code"], ns)


if __name__ == "__main__":
    main()

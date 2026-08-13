"""Working on a codebase from a cell: running the machine's own commands, and changing a
file in one place without rewriting it whole."""

import subprocess
from pathlib import Path


def sh(command, cwd=None):
    """Runs `command` through a shell and prints what it writes, as it arrives; returns
    the exit status.

    Printing rather than returning the output is the whole point of this function. The
    kernel keeps the protocol on a descriptor of its own and points file descriptor 1 at
    its log, so a subprocess started any other way writes where nobody reads: the cell
    reports a clean success and the build output is simply gone. Going through a pipe and
    a `print` puts it back in the one place the model actually sees, stderr included.
    """
    process = subprocess.Popen(
        command,
        shell=True,
        cwd=cwd,
        text=True,
        errors="replace",
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        for line in process.stdout:
            print(line, end="")
    except BaseException:
        # The interrupt reaches the kernel, never the child: without this a cancelled
        # cell leaves a build running against the same tree the next one works in.
        process.kill()
        raise
    finally:
        process.stdout.close()
    status = process.wait()
    if status != 0:
        print(f"[exit {status}]")
    return status


def edit(path, old, new):
    """Replaces `old` with `new` in `path`, exactly once.

    Refuses when the text is missing or appears more than once, because the failure worth
    guarding against is not a raised error — it is the edit that quietly also changed the
    three other lines that happened to match.
    """
    file = Path(path)
    text = file.read_text()
    found = text.count(old)
    if found == 0:
        raise ValueError(f"{file}: the text to replace is not in the file")
    if found > 1:
        raise ValueError(
            f"{file}: the text to replace appears {found} times; "
            "include enough around it to match once"
        )
    file.write_text(text.replace(old, new, 1))
    return f"{file}: replaced"

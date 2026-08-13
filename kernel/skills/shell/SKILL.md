---
name: shell
description: "Working on a codebase: running the project's tests, its build or any command with the output actually reaching you, since a subprocess started any other way writes where nobody reads and the cell reports a clean success; and changing one exact place in a file without rewriting it."
---

# shell

`import shell` in the cell: it holds two functions. Both are thin, and both exist because
the obvious way to do the same thing fails quietly.

`sh(command, cwd=None)` runs the command through a shell, prints what it writes as it
arrives, and returns the exit status. Printing rather than returning the output is the
whole point. The kernel keeps its protocol on a descriptor of its own and points file
descriptor 1 at its log, so a subprocess started any other way writes where nobody reads:
`subprocess.run("cargo test", shell=True)` comes back with a status and no error, the
cell reports a clean success, and the build output is simply gone. That is the trap — it
looks like it worked. stderr is folded into the same stream, so a compiler's diagnostics
arrive among its progress rather than in a second place you have to remember to look, and
a non-zero status prints `[exit N]` after the output, so a failure is visible even when
the command itself said nothing. The status is returned as well, so a cell can branch on
it. Pass `cwd=` to run somewhere else rather than prefixing `cd`.

An interrupted cell kills the child, so a cancelled build does not keep running against
the tree the next cell works in. A process you start yourself and abandon has no such
protection and outlives the cell that started it.

`edit(path, old, new)` replaces `old` with `new` exactly once, and raises when the text is
missing or appears more than once. The refusal is the feature: the failure worth guarding
against is not the edit that errors, it is the edit that succeeds and quietly also changed
the three other lines that happened to match — a bare `}`, a `return None`, an import. So
when the text is not unique, do not count the occurrences or loop over them. Widen `old`
with enough of the surrounding lines to name the one place you mean, and let the error
stand as proof that you named it.

The cell starts in the directory corpus was run from and stays there, so a project is read
and written with `pathlib` — `Path("src/main.rs").read_text()` rather than a shell `cat` —
and a relative path given to `sh` resolves against that same directory.

Run a project through its own toolchain rather than importing it here. This interpreter is
your workbench, not the project's environment: its packages are not the project's
packages, and a test that passed because you imported the module into this namespace has
proved nothing about the project. Read before you change, and check the change by running
what the project runs, whatever that is — `sh("cargo test")`, `sh("npm test")`,
`sh("pytest -q")`.

---
name: skill-creator
description: "Writing a skill of your own, so a procedure you had to work out once is read next time instead of rediscovered: a directory holding a SKILL.md and, where there is code, the package it describes. Read this before you write anything under a skills root, because a skill created in a root that did not exist when this session started imports only after `importlib.invalidate_caches()`, and a module you edit after importing it keeps the old code until it is reloaded — both fail in ways that look like the skill being wrong rather than new."
---

# skill-creator

Instructions only — there is nothing to import here.

Write a skill when the same work would otherwise be worked out twice: a library whose trap
you just spent three cells discovering, a procedure this project needs done one particular
way, a format someone here always wants. Do not write one for something you did once and
will not do again — a skills root full of guesses is a prompt nobody reads to the end.

## Where it goes

Two roots, and which one you pick is a question of who else it is for:

- `.corpus/skills/<name>/` — this project's. It belongs to the checkout, is committed with
  it, and is there for anyone who runs corpus in this directory.
- `~/.corpus/skills/<name>/` — this person's, on this machine, in every project.

A name found in both is taken from the project's, which is what lets a project override a
skill with its own version. The checkout's own `kernel/skills/` is corpus's shipped set:
add to it only when the work is changing corpus itself.

## The shape

```
.corpus/skills/tariffs/
├── SKILL.md          # required: frontmatter, then what a reader needs to know
├── __init__.py       # optional: the package a cell writes `import tariffs` for
└── requirements.txt  # optional: what pip installs for it, one to a line
```

The directory name is the skill's name and, where there is code, the import name with it.
Name it the way the skills beside it are named: the ordinary word for the thing it is
about, lowercase, one word wherever one word will do — `tariffs`, not `port-tariff-lookup`.
A skill with code cannot have a hyphen in its name at all, because `import` will not take
one; a skill that is only instructions may, and `skill-creator` is one.

The name must also not be the name of a package it imports: a skill directory called `docx`
hides python-docx, and the failure arrives as a missing `Document` rather than as a
collision.

The frontmatter is two lines, and the closing `---` matters:

```markdown
---
name: tariffs
description: "..."
---
```

`name` must equal the directory or the session says so on start. A skill without a
`description` is skipped outright — nothing would say what it is for, so nothing would ever
open it.

## The description is the routing

It is the only part that is always in a session's prompt. Everything else is read on
demand, by a model that decided to read it from this one sentence. So write what the skill
is for *and* what goes wrong without it, in the words someone would have in mind while
having the problem:

> "Reading and writing .xlsx and CSV. Read this before you write a formula into a sheet,
> because a formula this interpreter writes has no cached value and reads back as None."

not

> "Excel helper."

The second one is true and useless: it says nothing that would make a session open the file
at the moment it is needed.

## Using it in the session that wrote it

The skills roots are on the interpreter's path already, so a skill written under one that
was there when the session started imports at once:

```python
import tariffs
```

A root that did not exist at startup — the usual case, since `.corpus/skills` is created
the first time — is on the path but was cached as missing, so the import fails until that
cache is dropped:

```python
import importlib
importlib.invalidate_caches()
import tariffs
```

And a module you have already imported keeps the code you imported. Editing the file
changes nothing for this namespace until it is reloaded:

```python
importlib.reload(tariffs)
```

Each of the three fails as though the skill were wrong rather than new, which is why they
are written here rather than left to be worked out on the day it matters.

## Packages it needs

`requirements.txt` is read when a session starts, so writing it is what makes the next one
work, not this one. To use the package now, install it into the interpreter you are in:

```python
import shell, sys
shell.sh(f"{sys.executable} -m pip install python-pptx")
```

That is the same environment the next start would install into, so doing both is not
duplicated work — one line makes it usable now, the other makes it usable again.

## What this session's prompt says, and what the next one's will

The list of skills in your prompt was read when this session started. A skill you write now
is not in it, and you do not need it to be: you know what you just wrote. It is the *next*
session that gets it in the prompt, routed on the description, which is the whole point of
writing it down.

Agents you send off are in the same position — their prompt carries the list as it was at
the start. Point one at a new skill by naming the manifest in the task you give it: "read
`.corpus/skills/tariffs/SKILL.md`, then …".

## What belongs in the body

Write for a competent reader who has not met this problem. The house style, which the
skills beside this one follow:

- Open with what it is for and what it is *not* — the neighbouring skill that covers the
  other half of the job, named.
- The failure that stays quiet comes early. Anything that returns an empty string, a
  default, or a plausible wrong answer without raising is the reason the file exists at all.
- Ceilings are stated rather than discovered: what the subset does not cover, where the
  layout stops, what the wrapper deliberately does not do, and what to reach for instead.
- Show the one call that does the ordinary thing, and leave the library underneath
  reachable for everything else.
- No tutorial for what the library's own documentation covers. Write what its
  documentation will not tell you: what goes wrong here, on this machine, in this cell.

## Check it before you call it done

A skill that does not import is worse than no skill: the description promises a capability
and the promise fails in a later session, in front of the work rather than in front of you.
Before you say it is written, run it — import the module, call the function on real input,
and read the result. If it is instructions only, follow your own instructions once, in a
cell, and fix what you find. Then say what you wrote and where it is.

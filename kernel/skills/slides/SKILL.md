---
name: slides
description: "Producing a .pptx deck: `slides.deck(markdown, path)` cuts the same markdown a report is written in into slides, one per heading, with the tables and the charts under each. Read this before you promise a deck, because a slide does not reflow — everything under a heading is placed at full size and whatever does not fit runs off the bottom while the file saves without complaint."
---

# slides

`import slides` in the cell. It is a thin layer over `python-pptx`, and the library stays
reachable underneath. A PDF or a Word file is the `documents` skill; the graph
you put on a slide is `charts`.

Write the file into the working directory and hand it over with `send_user_file`. A deck is
delivered as a file, never as a description of one.

## The one call

```python
import slides
slides.deck("# Q3\n\n## Выручка\n\n- выросла на 12%\n\n![](chart.png)", "q3.pptx")
```

`deck(content, path="deck.pptx", title=None, template=None)` writes the file and returns
where it landed. The markdown is cut into slides like this:

- `#` opens the deck's title slide, and the paragraph under it becomes its subtitle.
- Every deeper heading — `##`, `###` — starts a new slide with that heading as its title.
- Lists and paragraphs under a heading become the lines of its body.
- A `|` table and an image on a line of its own each get a slide to themselves, under the
  same title, because both are laid out at the width of the slide.
- Material that arrives before any heading gets an untitled slide, named from `title` or
  the filename.

## Where the deck comes from

Write the report first and cut the deck from the same text. Two hand-written versions of
one set of numbers is how a deck comes to say something the report does not, and nothing
checks them against each other:

```python
report = build_the_markdown()
documents.pdf(report, "q3.pdf")
slides.deck(report, "q3.pptx")
```

Where a deck needs its own shape — a headline per slide rather than a section per slide —
write a second, shorter markdown for it rather than editing the .pptx afterwards.

## The failure that stays quiet

A slide does not reflow and does not shrink. Twenty bullets under one heading are all
placed, the ones past the bottom edge are simply outside the slide, and the file saves
without an error — you find out when someone presents it. So keep a heading to a handful of
lines and split the section when it grows, which is a change to your markdown rather than
to the deck.

A table is placed at the full height it was given, so a fifty-row table is unreadable
rather than paginated. Aggregate before you put a table on a slide, and leave the full one
in the report.

## What the markdown loses on the way

- `**bold**`, `*italic*` and `` `code` `` have their markers stripped and the words kept: a
  bullet is one run of text, and emphasis inside it is not something this places.
- Everything in a slide's body is a bullet, including what was a paragraph in the report.
- A link renders literally, brackets and all, exactly as in `documents`.
- Fenced code loses its fence and reflows, so a listing on a slide is one long line. Build
  that slide yourself.

## Fonts

There is no Latin-1 trap here. PowerPoint resolves fonts on the machine that opens the
file, so Cyrillic, Greek and CJK survive without anything being registered — unlike the PDF
side, where `documents.unicode_font()` exists precisely because reportlab embeds what it is
given.

## When the subset is not enough

`template=` inherits a .pptx: its slide size, its masters, its fonts and its colours. That
is how a deck comes out 16:9 or in a company's livery — the built-in default is
PowerPoint's own 4:3. The template is expected to keep the standard layout order (title
slide first, title-and-content second, title-only sixth), which the templates PowerPoint
ships and most corporate ones do.

Past that, build the presentation yourself:

```python
from pptx import Presentation
from pptx.util import Inches, Pt

show = Presentation()
slide = show.slides.add_slide(show.slide_layouts[6])      # blank
box = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(8), Inches(1))
box.text_frame.text = "Whatever the layout is the point of"
show.save("deck.pptx")
```

`add_picture`, `add_table`, `add_chart` and the placeholder API are all on `shapes`, and
speaker notes are `slide.notes_slide.notes_text_frame.text`.

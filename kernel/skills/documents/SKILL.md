---
name: documents
description: "Writing a PDF or a Word file, and assembling one from pages of others: `documents.pdf(markdown, path)` lays out a whole report in one call. Read this before you produce anything in Russian, Greek, Arabic or Chinese, because reportlab's built-in fonts are Latin-1 and those alphabets are written as empty boxes while nothing reports an error."
---

# documents

`import documents` in the cell. It is a thin layer over `reportlab` for the ordinary
report, and everything underneath it stays reachable. Getting text back *out* of a PDF is
the `scans` skill; putting a graph into one is `charts`; cutting the
same markdown into a deck is `slides`.

Write what you produce to a file in the working directory. A document is delivered as a
file, never as text in your answer.

## The one call

```python
import documents
documents.pdf("# Отчёт\n\nПервый абзац.", "report.pdf")   # -> 'report.pdf'
```

`pdf(content, path="report.pdf", title=None)` writes the file and returns where it landed.
`content` is either text in the markdown subset below or a list of flowables to lay out as
they are. `title` is the PDF's metadata title and defaults to the filename's stem. The page
is A4.

## What the markdown subset covers

`blocks(text)` is the parser `pdf` uses, and it understands `#` through `######` headings,
blank-line separated paragraphs, `-`/`*`/`+` and `1.` lists, `|` tables (the `|---|` rule
under the header is dropped and the first row is set bold), and `**bold**`, `*italic*` and
`` `code` `` within a line.

Everything else is read as prose, which is quieter than it sounds:

- A link keeps its brackets — `[text](url)` renders literally.
- A fenced block loses its fence, and its lines are joined into one paragraph like any
  other prose, so a code listing is reflowed into a run-on sentence — and a `#` comment
  inside it becomes a heading. Build listings yourself from
  `reportlab.platypus.Preformatted`.
- Line breaks inside a paragraph are the width of whoever typed it, not content, so a hard
  break does not survive. Separate paragraphs with a blank line.

## When the subset is not enough

`blocks(text)` returns the flowables rather than a file, so a document the subset cannot
express is assembled from those plus anything else reportlab offers:

```python
from reportlab.platypus import Image
documents.pdf(documents.blocks(text) + [Image("chart.png", width=400, height=200)],
              "report.pdf")
```

Reach for reportlab directly when the layout is the point. Deriving the ordinary layout by
hand when `pdf` already does it is wasted work.

## The font, and the failure that stays quiet

reportlab's built-in fonts are Latin-1. A heading in Cyrillic, Greek, Arabic or CJK set in
one of them comes out as filled boxes, the PDF is valid, and nothing in the pipeline
raises — the file only looks wrong once someone opens it.

`unicode_font()` registers the first system font that covers non-Latin text and returns its
name. `pdf` and `blocks` call it for you, so markdown needs nothing. Anything you build by
hand needs it passed explicitly:

```python
style = ParagraphStyle("Body", fontName=documents.unicode_font())
canvas.setFont(documents.unicode_font(), 12)
```

That covers a canvas, a `ParagraphStyle` you constructed, a `TableStyle` that sets `FONT`,
and every style in a `getSampleStyleSheet()` you fetched yourself: that call builds a fresh
sheet each time, so the one `blocks` fixes is its own and yours arrives on Helvetica. Of its
own sheet `blocks` leaves only `Code` on Courier, to keep it monospaced, which means
non-Latin text set in that style is boxes. `blocks(text, font=...)` takes a font name if you
registered your own face. On a machine with no usable font, `unicode_font()` raises
`LookupError` rather than writing a broken file.

`font_file()` returns the same face as a path, for a library that wants a file rather than a
registered name — which is how `charts` points matplotlib at it.

## Ceilings worth knowing before you hit them

- There is no italic face. `<i>` and `*italic*` map back to the regular weight, so emphasis
  written that way renders as no emphasis at all. Bold works wherever the machine ships a
  bold file beside the regular one.
- Inline `` `code` `` in a non-ASCII word drops the Courier face and renders as prose,
  because Courier would render it as boxes. The alphabet is kept, the monospace is not.
- Table columns are even, and their width is measured against the A4 page `pdf` builds. A
  table whose first column is one word wide will look it; build the `Table` yourself with
  explicit `colWidths` when that matters.

## Assembling from pages that already exist

`pypdf.PdfWriter` does the file-level work: `append(path)` concatenates whole documents,
`add_page(reader.pages[n])` takes single pages, and `write(path)` saves. Rotation,
encryption and form filling live on the same writer. None of it re-lays-out anything, so
every page keeps its own fonts and embedded resources — which also means the Latin-1
question was settled by whoever wrote each page, not by you.

## Word

`python-docx` is installed here and is a library, not a wrapper: import and use it
directly, it does not go through `documents`. Word carries no font problem of its own,
because Word resolves fonts on the machine that opens the file — the Latin-1 trap above is
reportlab's alone.

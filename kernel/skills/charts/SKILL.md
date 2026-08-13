---
name: charts
description: "Drawing a graph with matplotlib and putting it in a report. Read this before the first figure of a session: there is no display on this machine, so a default backend can hang the cell, figures left open accumulate across cells because the namespace persists, and Japanese or Chinese labels render as boxes in the default face."
---

# charts

Instructions only — there is nothing to import here. `matplotlib` is installed and you use
it directly. Laying the surrounding report out is the `documents` skill.

## The two lines that go before the first figure

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

There is no display here. `use("Agg")` selects the file-writing backend outright rather
than letting matplotlib search for a windowing system it will not find. It has to come
before `pyplot` is imported; afterwards it is too late and the call is ignored.

## The namespace persists, and so do figures

`plt.subplots()` registers the figure with pyplot, which holds it until something closes
it. Because this interpreter keeps its variables between cells, figures accumulate for the
whole session — a loop over thirty regions leaves thirty open, and matplotlib starts
warning about it around twenty. Close each one as you finish with it:

```python
fig, ax = plt.subplots(figsize=(5, 2.5))
ax.bar(["Север", "Юг"], [40, 60])
ax.set_title("Выручка по регионам")
fig.savefig("chart.png", dpi=150, bbox_inches="tight")
plt.close(fig)
```

`bbox_inches="tight"` crops the generous default margin, which otherwise reads as a
mispositioned image once the PNG is placed in a document.

## The font, one library over

matplotlib's default face is DejaVu Sans, which covers Cyrillic and Greek but has no CJK at
all, so Japanese or Chinese labels come out as boxes. It is louder about this than
reportlab — you get a `Glyph ... missing from font` warning per character rather than
silence — but a warning printed by a cell that also printed its output is a warning you
will scroll past.

Point it at the same face the documents use, once per session:

```python
import documents
from matplotlib import font_manager

path = documents.font_file()
font_manager.fontManager.addfont(str(path))
plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(path)).get_name()
```

`documents.font_file()` returns the first system font that covers non-Latin text, or `None`
on a machine that has none. Setting `rcParams` once means every later figure inherits it.

## Into the report

A chart reaches a PDF as a file, through a flowable:

```python
from reportlab.platypus import Image
documents.pdf(documents.blocks(text) + [Image("chart.png", width=400, height=200)],
              "report.pdf")
```

`width` and `height` are points, and they are not read from the PNG: give the aspect ratio
you saved at or the image arrives stretched. A figure saved at `figsize=(5, 2.5)` is twice
as wide as it is tall, whatever its pixel dimensions.

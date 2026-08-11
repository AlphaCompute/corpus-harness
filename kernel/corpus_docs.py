"""Fonts for documents, because reportlab ships Latin-1 built-ins: a heading in Cyrillic,
Greek or anything else comes out as filled boxes unless a real TTF is registered first, and
nothing in the pipeline complains — the file only looks wrong once someone opens it."""

import re
from pathlib import Path

# Whole-Unicode faces first, then the DejaVu that every Linux distribution puts somewhere
# slightly different.
CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
)


# Tried against the regular file's own name, because a foundry ships the bold weight
# beside it and only the punctuation between the two words varies.
BOLD_SUFFIXES = (" Bold", "-Bold", "Bold", "bd")


def _bold_for(regular):
    """The bold weight to set beside a font file, or None where the machine has none.

    The face picked for its coverage is not always shipped in two weights — the widest
    one on a Mac is not — so a bold belonging to another candidate is taken rather than
    leaving emphasis to render as no emphasis at all.
    """
    for base in (regular, *(Path(path) for path in CANDIDATES)):
        for suffix in BOLD_SUFFIXES:
            bold = base.with_name(base.stem + suffix + base.suffix)
            if bold.exists():
                return bold
    return None


def unicode_font(name="Unicode"):
    """Registers the first system font that covers non-Latin text and returns its name,
    for `fontName=` on a canvas or a paragraph style. `<b>` inside a paragraph picks up
    the bold weight wherever the machine ships one beside the regular file.

    ponytail: no italic. `<i>` maps back to the regular face, so emphasis written that
    way is silently flat; the upgrade is a second lookup for the oblique file.
    """
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    if name in pdfmetrics.getRegisteredFontNames():
        return name
    for path in CANDIDATES:
        path = Path(path)
        if not path.exists():
            continue
        pdfmetrics.registerFont(TTFont(name, str(path)))
        bold = _bold_for(path)
        if bold is not None:
            pdfmetrics.registerFont(TTFont(f"{name}-Bold", str(bold)))
        # Mapped even where there is no bold file: an unmapped family sends `<b>` to
        # Helvetica-Bold, which is Latin-1 and loses the alphabet this was chosen for.
        pdfmetrics.registerFontFamily(
            name,
            normal=name,
            bold=f"{name}-Bold" if bold is not None else name,
            italic=name,
            boldItalic=f"{name}-Bold" if bold is not None else name,
        )
        return name
    raise LookupError(
        "no unicode font on this machine; register a .ttf yourself with "
        "reportlab.pdfbase.pdfmetrics.registerFont(TTFont(name, path))"
    )


_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_BULLET = re.compile(r"^\s*[-*+]\s+(.*)$")
_NUMBERED = re.compile(r"^\s*\d+[.)]\s+(.*)$")
# The `|---|:--|` row under a table's header, which is a rule rather than data.
_RULE = re.compile(r"^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$")


def pdf(content, path="report.pdf", title=None):
    """Writes a PDF and returns where it landed.

    `content` is either text in the markdown subset `blocks` understands, or flowables
    to lay out as they are:

        corpus_docs.pdf("# Отчёт\\n\\nПервый абзац.", "report.pdf")
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Spacer

    story = blocks(content) if isinstance(content, str) else list(content)
    SimpleDocTemplate(
        str(path), pagesize=A4, title=title or Path(path).stem
    ).build(story or [Spacer(1, 1)])
    return str(path)


def blocks(text, font=None):
    """The flowables `pdf` would lay out for this text, so a document that needs
    something the subset does not cover can be assembled from these plus anything else
    reportlab offers: `pdf(blocks(text) + [Image('chart.png')], 'report.pdf')`.

    Understood: `#` headings, blank-line separated paragraphs, `-` and `1.` lists,
    `|` tables, and `**bold**`, `*italic*` and `` `code` `` within a line.

    ponytail: everything else is read as prose, so a link keeps its brackets and a
    fenced block loses its fence. Each is a branch in the loop below on the day a
    document needs it.
    """
    from reportlab.platypus import ListFlowable, ListItem, Paragraph

    styles = _styles(font or unicode_font())
    out = []
    lines = text.splitlines()
    at = 0
    while at < len(lines):
        line = lines[at]
        if not line.strip():
            at += 1
            continue

        heading = _HEADING.match(line)
        if heading:
            level = min(len(heading.group(1)), 6)
            body = _inline(heading.group(2))
            out.append(Paragraph(f"<b>{body}</b>", styles[f"Heading{level}"]))
            at += 1
        elif _is_row(line):
            rows, at = _take(lines, at, _is_row)
            out.append(_table(rows, styles))
        elif _item(line) is not None:
            numbered = _NUMBERED.match(line) is not None
            raw, at = _take(lines, at, lambda l: _item(l) is not None)
            items = [
                ListItem(Paragraph(_inline(_item(line)), styles["BodyText"]))
                for line in raw
            ]
            out.append(
                ListFlowable(
                    items,
                    bulletType="1" if numbered else "bullet",
                    bulletFontName=styles["BodyText"].fontName,
                    leftIndent=18,
                )
            )
        else:
            # A paragraph runs until something else starts, and its line breaks are the
            # width of whoever typed it rather than anything the reader should see.
            raw, at = _take(lines, at, lambda l: l.strip() and not _opens_block(l))
            body = " ".join(piece.strip() for piece in raw)
            out.append(Paragraph(_inline(body), styles["BodyText"]))
    return out


def _styles(font):
    from reportlab.lib.styles import getSampleStyleSheet

    sheet = getSampleStyleSheet()
    for name, style in sheet.byName.items():
        # `Code` is monospaced on purpose, and no substitute for it is guaranteed here.
        if name == "Code":
            continue
        style.fontName = font
        if getattr(style, "bulletFontName", None):
            style.bulletFontName = font
    return sheet


def _take(lines, at, keep):
    """The run of lines from `at` that `keep` accepts, and where it ended."""
    start = at
    while at < len(lines) and keep(lines[at]):
        at += 1
    return lines[start:at], at


def _item(line):
    """What a list item says, or None when the line is not one."""
    match = _BULLET.match(line) or _NUMBERED.match(line)
    return match.group(1) if match else None


def _is_row(line):
    return line.lstrip().startswith("|")


def _opens_block(line):
    return (
        _HEADING.match(line) is not None or _is_row(line) or _item(line) is not None
    )


def _table(rows, styles):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

    cells = []
    for row in rows:
        if _RULE.match(row):
            continue
        parts = [part.strip() for part in row.strip().strip("|").split("|")]
        # The header carries the weight, so it says so rather than relying on the fill.
        emphasis = "<b>{}</b>" if not cells else "{}"
        cells.append(
            [Paragraph(emphasis.format(_inline(part)), styles["BodyText"]) for part in parts]
        )
    if not cells:
        return Spacer(1, 1)

    # reportlab wants a rectangle; a row that is short of one is the writer's slip.
    columns = max(len(row) for row in cells)
    for row in cells:
        row.extend(Paragraph("", styles["BodyText"]) for _ in range(columns - len(row)))

    # ponytail: even columns, and the page they are measured against is the A4 that
    # `pdf` builds. Measuring the content instead is the upgrade, and a table whose
    # first column is one word wide is what will ask for it.
    width = (A4[0] - 2 * inch) / columns
    table = Table(cells, colWidths=[width] * columns, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def _inline(text):
    """Markdown emphasis as the mini-HTML a reportlab paragraph reads. Escaped first,
    because a `<` that came from the text is content and must not become a tag."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"`([^`]+)`", _code, text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return re.sub(r"\*([^*\n]+)\*", r"<i>\1</i>", text)


def _code(match):
    body = match.group(1)
    # Courier is Latin-1, so a path or an identifier is safe in it and a Cyrillic word
    # would come out as boxes. That one reads as prose instead of not at all.
    return f'<font face="Courier">{body}</font>' if body.isascii() else body

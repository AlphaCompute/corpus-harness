---
name: scans
description: "Getting the text out of a PDF someone handed you, including one that is photographs of pages rather than text. Read this before you summarise or search a PDF you did not write, because extraction from a scan returns an empty string rather than an error, and an empty string reads downstream exactly like a document with nothing in it."
---

# scans

Instructions only — there is nothing to import here. `pypdf` and `pymupdf` are installed
and you use them directly. Writing a PDF is the `documents` skill; finding which
of a folder full of them a question is about, rather than reading one, is `search`.

## The failure to rule out first

`pypdf.PdfReader(path).pages[n].extract_text()` is the ordinary way in, and on a scanned
document it returns `""`. That is not a bug: there is no text in the file to extract, only
a picture of text. Nothing raises, and the empty string flows onward as though the page
were blank. A summary built on it will be confidently about nothing.

So check what came back before you build on it, and when it is empty, ask the page what it
does contain:

```python
import pymupdf

page = pymupdf.open("report.pdf")[0]
text = page.get_text()
if not text.strip() and page.get_images():
    text = page.get_text(textpage=page.get_textpage_ocr(dpi=200))
```

Nothing extracted beside at least one image is a scan. Do this per page rather than per
file: a report is very often typeset text with a photographed appendix, and the file-level
answer is "some text, therefore fine".

## OCR is a program, not a package

`get_textpage_ocr` shells out to tesseract, which is installed on the machine or is not —
`pip` cannot supply it, and nothing in this skill's requirements can. Where it is missing
the call raises rather than quietly returning nothing, which is the right way round. Say
plainly that the document needs OCR and the machine has no tesseract, rather than
delivering a summary of the pages that happened to have a text layer.

`dpi=200` is a reasonable floor. Below it, small type is lost silently — the OCR returns
fewer words rather than an error.

## Why pymupdf rather than pypdf for reading

Both read a text layer. `pymupdf` also gives you what you need to judge the result:
`page.get_images()` for the scan test above, `page.find_tables()` for tabular pages,
`page.get_pixmap()` to render a page as an image, and per-span font, size and position
detail that `pypdf` does not expose. Use `pypdf` for assembling files, `pymupdf` for
interrogating them.

The import is `pymupdf`. The old `fitz` name still works and prints a deprecation warning.

## Reading in bulk

Text extracted from a long document is material, not context: assign it, print a slice to
see its shape, and write the code that handles the rest against what you saw. A hundred
pages chunked and mapped through `llm_batch` is one round of calls; a hundred pages read
through your own context is a hundred turns and a worse answer.

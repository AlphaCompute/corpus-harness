---
name: search
description: "Finding which pages of a pile of documents a question is about — `search.index(dir).find(question)` ranks the chunks of PDFs, Word files and text by BM25 and hands back path, page and snippet. Read this before you search a folder you cannot read through, because the match is on words and not on meaning: a question asked in other words than the document uses scores nothing, and an empty result reads exactly like an absent fact."
---

# search

`import search` in the cell. It is a small layer over duckdb's full-text index, and both
duckdb and the extractors stay reachable underneath. Structured files queried as tables are
the `queries` skill; getting text out of one document, scan included, is
`scans`; keeping a found claim attached to its source is `citations`.

## The two calls

```python
import search

found = search.index("reports/")           # a directory, a glob, or a list of paths
found                                        # <index of 412 chunks from 37 files>
for hit in found.find("страховой резерв", k=5):
    print(hit["score"], hit["path"], hit["page"], hit["snippet"])
```

`index(paths, stemmer="porter")` reads `.pdf` page by page, `.docx` and text files in
pieces of about 1500 characters, and returns an index held in memory. `find(query, k=10)`
returns the best chunks first, each as `{path, page, score, snippet, text}`. `page` is the
page number for a PDF and the ordinal of the chunk for everything else.

Read `snippet` and print `snippet`. `text` is the whole chunk — the unit a citation points
at, and the thing to pass on to `llm_batch` — and ten of them printed is a page of your
context spent on material you have not decided to use yet.

## What it does not do

BM25 scores words, not meaning. A question in the document's own vocabulary finds it; the
same question in other words finds nothing, and nothing looks identical to a corpus that
does not contain the fact. So:

- Search the words the document would use, not the words the question came in. Try two or
  three phrasings and union the results — each `find` is milliseconds once the index is
  built.
- When the wording is genuinely unknown, use this to narrow rather than to answer: take the
  top few dozen chunks, map them through `llm_batch` with the real question, and aggregate
  in code. That is the reading a model does well and a word index cannot do at all.
- An empty result is worth reporting as an empty result. "Not found in the corpus by
  search" and "not in the corpus" are different claims, and only the first one is yours.

## Language

The default stemmer is English. Russian, German and the rest of the snowball list are named
by language, and mixed-language material is best matched as written:

```python
search.index("отчёты/", stemmer="russian")
search.index("mixed/", stemmer="none")
```

An English stemmer over Russian text does not raise; it merely stops matching inflections,
so a query in the nominative misses the same word in the genitive. Nothing says so — the
result is a shorter list.

## The files that gave nothing

```python
found.empty      # read, and held no text at all
found.skipped    # not a kind this reads
```

A PDF that is photographs of pages extracts as an empty string rather than an error, so it
enters the index as nothing and is never a hit. `empty` is where those land, and it is
almost always scans: put them through the OCR path in `scans` and index the text you get
back. Check both lists before you conclude a corpus does not mention something — an index
built over a tenth of a directory answers confidently about the other nine tenths.

`.pptx`, `.xlsx`, images and archives are in `skipped`. Spreadsheets are usually the
`queries` skill's job rather than this one's.

## The index is a variable, not a file

It lives in the kernel for as long as the session does, so build it once and search it many
times. A kernel restart costs the extraction again — minutes over thousands of pages —
which is a reason to build it in its own cell and keep the name, not a reason to rebuild it
per question.

The full-text extension is installed into duckdb on first use, which needs the network
once; it is cached under `~/.duckdb/extensions` afterwards. On a machine with neither, the
call says so plainly rather than searching nothing.

## Under it

The index is an ordinary duckdb connection over a `passages(id, path, page, text)` table,
reachable as `found.db` — so a filter the API does not offer is SQL you write yourself:

```python
found.db.execute(
    "SELECT path, page, text FROM passages WHERE path LIKE '%2024%' AND text ILIKE '%резерв%'"
).fetchall()
```

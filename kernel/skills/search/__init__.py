"""Finding the few places in a pile of documents that a question is actually about, so what
enters the cell is those places rather than the pile."""

import glob
import re
from pathlib import Path

# A page is the unit for a PDF because that is what a reader can be pointed at. Everything
# else is cut to about this many characters, on a paragraph boundary where there is one
# nearby: small enough that a hit names a place rather than a file, large enough that the
# sentence around the hit is still there.
CHUNK = 1500

# What is read, and how. Anything else is skipped and counted — a corpus is usually a
# directory of everything, and silently indexing a tenth of it is the failure here.
TEXT = (".txt", ".md", ".rst", ".log", ".csv", ".json", ".yaml", ".yml", ".html", ".xml")

_WORD = re.compile(r"\w{3,}", re.UNICODE)


def index(paths, stemmer="porter"):
    """Reads what `paths` names and returns an [`Index`] to search.

    `paths` is a directory (read through), a glob (`"reports/**/*.pdf"`), or an iterable of
    paths. `stemmer` is duckdb's — `"porter"` for English, `"russian"`, `"german"` and the
    rest of the snowball list for their languages, `"none"` to match words as written,
    which is what mixed-language material wants.

        found = search.index("reports/")
        found.find("страховой резерв")
    """
    if not (isinstance(stemmer, str) and stemmer.isalpha()):
        raise ValueError(f"{stemmer!r} is not a stemmer name")

    rows, empty, skipped = [], [], []
    for path in _files(paths):
        if path.suffix.lower() not in (".pdf", ".docx", *TEXT):
            skipped.append(path)
            continue
        pieces = [(page, text) for page, text in _read(path) if text.strip()]
        # A file that yielded nothing is not a file with nothing in it — see the skill.
        if not pieces:
            empty.append(path)
        for page, text in pieces:
            rows.append((len(rows), str(path), page, text))
    return Index(rows, stemmer, empty, skipped)


class Index:
    """A duckdb full-text index over the chunks, held in memory for as long as the session
    keeps this object.

    ponytail: in memory, so a kernel restart costs the extraction again — minutes on a
    corpus of thousands of pages. The upgrade is a duckdb file and a check on whether the
    files have changed since it was written.
    """

    def __init__(self, rows, stemmer, empty, skipped):
        import duckdb

        self.files = len({row[1] for row in rows})
        self.chunks = len(rows)
        #: Read and found to hold no text at all. Almost always scans — see `scans`.
        self.empty = empty
        #: Not a kind this reads. Listed so a corpus that is mostly `.pptx` says so.
        self.skipped = skipped

        self.db = duckdb.connect()
        try:
            self.db.execute("INSTALL fts; LOAD fts")
        except Exception as error:
            raise RuntimeError(
                "duckdb's full-text extension is not on this machine, and installing one "
                "needs the network. Fall back to `re` over the chunks, or install it where "
                "there is a connection: it is cached under ~/.duckdb/extensions afterwards."
            ) from error
        self.db.execute("CREATE TABLE passages (id BIGINT, path VARCHAR, page INTEGER, text VARCHAR)")
        # Nothing to index is not an error: an empty corpus answers every query with
        # nothing, which is the truth about it. duckdb refuses both calls on no rows.
        if rows:
            self.db.executemany("INSERT INTO passages VALUES (?, ?, ?, ?)", rows)
            self.db.execute(
                f"PRAGMA create_fts_index('passages', 'id', 'text', stemmer='{stemmer}', overwrite=1)"
            )

    def find(self, query, k=10):
        """The `k` best-matching chunks, each as `{path, page, score, snippet, text}` and
        best first. `page` is the page for a PDF and the ordinal of the chunk for anything
        else. Nothing matched is an empty list rather than a row of zero scores."""
        if not self.chunks:
            return []
        rows = self.db.execute(
            "SELECT path, page, text, score FROM ("
            "  SELECT *, fts_main_passages.match_bm25(id, ?) AS score FROM passages"
            ") WHERE score IS NOT NULL ORDER BY score DESC LIMIT ?",
            [query, k],
        ).fetchall()
        return [
            {
                "path": path,
                "page": page,
                "score": round(score, 3),
                "snippet": _snippet(text, query),
                "text": text,
            }
            for path, page, text, score in rows
        ]

    def __repr__(self):
        return f"<index of {self.chunks} chunks from {self.files} files>"


def _files(paths):
    if isinstance(paths, (str, Path)):
        if Path(paths).is_dir():
            return sorted(path for path in Path(paths).rglob("*") if path.is_file())
        return sorted(
            Path(hit) for hit in glob.glob(str(paths), recursive=True) if Path(hit).is_file()
        )
    return [Path(path) for path in paths]


def _read(path):
    """`(page, text)` for one file."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        import pymupdf

        with pymupdf.open(path) as document:
            # Materialised inside the `with`, because the pages go away with the document.
            return [(number, page.get_text()) for number, page in enumerate(document, 1)]
    if suffix == ".docx":
        import docx

        text = "\n\n".join(para.text for para in docx.Document(str(path)).paragraphs)
        return _cut(text)
    return _cut(path.read_text(encoding="utf-8", errors="replace"))


def _cut(text):
    """The text in pieces of about `CHUNK`, broken between paragraphs where it can be."""
    pieces, piece = [], ""
    for paragraph in re.split(r"\n\s*\n", text):
        if piece and len(piece) + len(paragraph) > CHUNK:
            pieces.append(piece)
            piece = ""
        # A paragraph longer than a chunk on its own — a wrapped-once wall of text — is cut
        # where it is rather than left as one piece the size of the file.
        while len(paragraph) > CHUNK:
            pieces.append(paragraph[:CHUNK])
            paragraph = paragraph[CHUNK:]
        piece = f"{piece}\n\n{paragraph}" if piece else paragraph
    if piece.strip():
        pieces.append(piece)
    return list(enumerate(pieces, 1))


def _snippet(text, query, room=160):
    """The window around the first query word that appears, so a hit can be read without
    printing the whole chunk."""
    words = [word.casefold() for word in _WORD.findall(query)]
    body = text.casefold()
    at = min((found for found in (body.find(word) for word in words) if found >= 0), default=-1)
    start = max(at - room // 2, 0) if at >= 0 else 0
    window = " ".join(text[start : start + room].split())
    return f"{'…' if start else ''}{window}{'…' if start + room < len(text) else ''}"

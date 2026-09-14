"""Markdown-aware document chunking.

Splits on headings first, then packs paragraphs up to a size budget. Heading-first
matters here: the corpus is written as short topical sections, so a heading
boundary is almost always a genuine topic boundary. Splitting purely on character
count would cut a treatment table away from the pest it belongs to and depress
contextual precision in the Phase 5 retrieval metrics.

Each chunk keeps its heading trail, which is prepended to the embedded text so
that a chunk retrieved on its own still says what it is about.
"""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")

DEFAULT_MAX_CHARS = 1200
DEFAULT_MIN_CHARS = 120


@dataclass(frozen=True)
class Document:
    """A source document before chunking."""

    id: str
    title: str
    source: str
    text: str


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit of text."""

    id: str
    doc_id: str
    doc_title: str
    ordinal: int
    heading: str
    text: str
    #: Tags parsed from the document front matter, used as LanceDB filters.
    tags: list[str] = field(default_factory=list)

    @property
    def embed_text(self) -> str:
        """Text actually handed to the embedding model.

        The heading trail is prepended so an isolated chunk carries its own topic.
        """
        return f"{self.doc_title} - {self.heading}\n\n{self.text}" if self.heading else self.text


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def parse_document(path: Path) -> Document:
    """Read a corpus markdown file.

    The first level-1 heading becomes the title; the filename stem becomes the id.
    """
    text = path.read_text(encoding="utf-8").strip()
    title = path.stem.replace("_", " ").title()

    for line in text.splitlines():
        match = HEADING_RE.match(line)
        if match and len(match.group(1)) == 1:
            title = match.group(2).strip()
            break

    return Document(id=f"doc_{path.stem}", title=title, source=path.name, text=text)


def _sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into ``(heading, body)`` pairs."""
    sections: list[tuple[str, str]] = []
    heading = ""
    buffer: list[str] = []

    for line in text.splitlines():
        match = HEADING_RE.match(line)
        if match:
            if buffer:
                sections.append((heading, "\n".join(buffer).strip()))
                buffer = []
            # A level-1 heading is the document title, not a section header.
            heading = "" if len(match.group(1)) == 1 else match.group(2).strip()
        else:
            buffer.append(line)

    if buffer:
        sections.append((heading, "\n".join(buffer).strip()))

    return [(h, b) for h, b in sections if b]


def _pack(body: str, max_chars: int) -> list[str]:
    """Pack paragraphs into pieces no longer than ``max_chars``."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    pieces: list[str] = []
    current: list[str] = []
    size = 0

    for paragraph in paragraphs:
        # A single oversized paragraph is emitted alone rather than split
        # mid-sentence; the corpus is authored to avoid this.
        if size and size + len(paragraph) > max_chars:
            pieces.append("\n\n".join(current))
            current, size = [], 0
        current.append(paragraph)
        size += len(paragraph) + 2

    if current:
        pieces.append("\n\n".join(current))

    return pieces


def chunk_document(
    document: Document,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> list[Chunk]:
    """Split a document into retrievable chunks.

    Sections shorter than ``min_chars`` are merged into the previous chunk rather
    than emitted alone - a two-line stub retrieves badly and pollutes precision.
    """
    chunks: list[Chunk] = []

    for heading, body in _sections(document.text):
        for piece in _pack(body, max_chars):
            if chunks and len(piece) < min_chars:
                previous = chunks.pop()
                chunks.append(
                    Chunk(
                        id=previous.id,
                        doc_id=previous.doc_id,
                        doc_title=previous.doc_title,
                        ordinal=previous.ordinal,
                        heading=previous.heading,
                        text=f"{previous.text}\n\n{piece}",
                    )
                )
                continue

            ordinal = len(chunks)
            digest = hashlib.sha1(f"{document.id}:{ordinal}:{piece}".encode()).hexdigest()[:10]
            chunks.append(
                Chunk(
                    id=f"chunk_{_slug(document.id)}_{ordinal:02d}_{digest}",
                    doc_id=document.id,
                    doc_title=document.title,
                    ordinal=ordinal,
                    heading=heading,
                    text=piece,
                )
            )

    return chunks


def chunk_corpus(
    corpus_dir: Path,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> tuple[list[Document], list[Chunk]]:
    """Parse and chunk every markdown file in ``corpus_dir``."""
    documents: list[Document] = []
    chunks: list[Chunk] = []

    for path in sorted(corpus_dir.glob("*.md")):
        document = parse_document(path)
        documents.append(document)
        chunks.extend(chunk_document(document, max_chars=max_chars))

    return documents, chunks

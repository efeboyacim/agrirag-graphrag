"""Chunking behaviour."""

from pathlib import Path

from agrirag.ingestion.chunker import (
    Document,
    chunk_corpus,
    chunk_document,
    parse_document,
)

CORPUS = Path("data/corpus")


def _doc(text: str) -> Document:
    return Document(id="doc_test", title="Test Doc", source="test.md", text=text)


def test_level_one_heading_becomes_the_title_not_a_section() -> None:
    """The document title should not be duplicated as a section heading."""
    chunks = chunk_document(
        _doc("# Real Title\n\n" + "Body paragraph. " * 20 + "\n\n## Section\n\n" + "More. " * 40)
    )

    headings = [c.heading for c in chunks]
    assert "Real Title" not in headings
    assert "Section" in headings


def test_sections_become_separate_chunks() -> None:
    text = "# Doc\n\n## Alpha\n\n" + "Alpha body. " * 30 + "\n\n## Beta\n\n" + "Beta body. " * 30
    chunks = chunk_document(_doc(text))

    assert [c.heading for c in chunks] == ["Alpha", "Beta"]
    assert "Alpha body" in chunks[0].text
    assert "Beta body" in chunks[1].text


def test_a_long_section_is_split_at_a_paragraph_boundary() -> None:
    paragraph = "Sentence content here. " * 25  # ~575 chars
    text = "# Doc\n\n## Long\n\n" + "\n\n".join([paragraph] * 4)

    chunks = chunk_document(_doc(text), max_chars=800)

    assert len(chunks) > 1
    # Splitting must never land mid-paragraph.
    for chunk in chunks:
        assert chunk.text.strip().startswith("Sentence content here.")


def test_a_short_section_is_merged_into_the_previous_chunk() -> None:
    """A two-line stub retrieves badly on its own, so it is folded backwards."""
    text = "# Doc\n\n## Big\n\n" + "Body text. " * 40 + "\n\n## Tiny\n\nShort note.\n"

    chunks = chunk_document(_doc(text), min_chars=120)

    assert len(chunks) == 1
    assert "Short note." in chunks[0].text


def test_embed_text_carries_the_title_and_heading() -> None:
    """A chunk retrieved in isolation must still state what it is about."""
    chunks = chunk_document(_doc("# Doc\n\n## Section\n\n" + "Body. " * 40))

    assert chunks[0].embed_text.startswith("Test Doc - Section")
    assert chunks[0].text.startswith("Body.")


def test_chunk_ids_are_deterministic() -> None:
    """Re-ingestion must MERGE onto existing chunks rather than duplicate them."""
    text = "# Doc\n\n## Section\n\n" + "Body. " * 40
    first = chunk_document(_doc(text))
    second = chunk_document(_doc(text))

    assert [c.id for c in first] == [c.id for c in second]


def test_chunk_id_changes_when_the_text_changes() -> None:
    """An edited document must not silently keep a stale chunk id."""
    base = "# Doc\n\n## Section\n\n" + "Body. " * 40
    edited = base + "extra sentence."

    assert chunk_document(_doc(base))[0].id != chunk_document(_doc(edited))[0].id


def test_parse_document_reads_the_title_from_the_first_h1(tmp_path: Path) -> None:
    path = tmp_path / "some_file.md"
    path.write_text("# Proper Title\n\nBody.\n", encoding="utf-8")

    document = parse_document(path)

    assert document.title == "Proper Title"
    assert document.id == "doc_some_file"
    assert document.source == "some_file.md"


def test_parse_document_falls_back_to_the_filename(tmp_path: Path) -> None:
    path = tmp_path / "no_heading_here.md"
    path.write_text("Just body text.\n", encoding="utf-8")

    assert parse_document(path).title == "No Heading Here"


def test_real_corpus_chunks_within_bounds() -> None:
    """Guards the corpus itself: an over-long section would degrade precision."""
    documents, chunks = chunk_corpus(CORPUS)

    assert len(documents) >= 20
    assert chunks
    assert len({c.id for c in chunks}) == len(chunks)
    assert all(len(c.text) <= 1200 for c in chunks)
    assert all(c.doc_id.startswith("doc_") for c in chunks)

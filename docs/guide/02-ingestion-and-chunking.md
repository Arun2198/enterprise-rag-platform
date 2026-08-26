# Chapter 2: Document Ingestion & Chunking

This chapter covers the first two stages of the pipeline: getting text out of a source file
(`src/ingestion/`), and splitting that text into the small pieces the rest of the system actually
searches over (`src/rag/chunking/`).

## 1. Why chunking exists at all

You might expect a RAG system to just embed and search whole documents. That doesn't work well in
practice, for two reasons:

1. **Precision.** A 40-page PDF embedded as one vector produces a single, blurry average of
   everything the document talks about. A question about one specific paragraph on page 12 won't
   match that average well. Embed small, focused pieces instead, and a question matches the piece
   that's actually about it.
2. **Context budget.** An LLM's prompt has a token limit (Chapter 0, section 2). You can't hand it
   a whole document library for every question — you can only afford to hand it the few passages
   that are actually relevant, which retrieval (Chapter 4) finds by comparing against these small
   pieces, not whole documents.

So: split every document into **chunks** — pieces small enough to search precisely and fit
several into a prompt, but large enough to still contain a complete thought.

## 2. Ingestion: `IngestionPipeline.ingest_file()`

Located in [`src/ingestion/ingestion_pipeline.py`](../../src/ingestion/ingestion_pipeline.py).
Given a file path, it:

1. Asks `ParserFactory.get_parser(file_path)` for a parser, chosen purely by file extension
   (`.pdf` → `PDFParser`, `.docx` → `DOCXParser`, `.md`/`.markdown` → `MarkdownParser`; anything
   else returns an `UNSUPPORTED_FILE_TYPE` error).
2. Calls that parser's `.parse(file_path)`, which reads the file and returns a `Document`
   (defined in `src/ingestion/contracts/document.py`) — an object holding `document_id`,
   `source`, `document_type`, and the extracted `content` as one string, plus optional
   `owner`/timestamps/`metadata`.
3. Runs the extracted content through `TextCleaner.clean()` (`src/ingestion/cleaners/
   text_cleaner.py`) to normalize whitespace (collapsing repeated blank lines and spacing
   artifacts PDF extraction tends to leave behind).

### The `Result[T]` pattern — no exceptions for expected failures

Every parser and pipeline stage returns a `Result[T]` (`src/ingestion/contracts/result.py`)
instead of raising an exception for anything that's a normal, expected failure mode (unsupported
file type, corrupt PDF, empty document):

```python
class Error(BaseModel):
    code: str
    message: str

class Result(BaseModel, Generic[T]):
    success: bool
    data: T | None = None
    error: Error | None = None
```

The reasoning: a malformed uploaded file is not a programming bug, it's an expected input the
system needs to handle gracefully. Using return values instead of exceptions means every caller
*must* check `.success` to get at `.data` — the type system won't let you accidentally use `None`
data as if it succeeded, and the caller controls exactly what happens on failure instead of an
exception unwinding the stack past code that could have handled it cleanly. This is why an
`/ingest` call with a bad file doesn't crash the whole request — `RAGService.ingest()` (walked
through below) just appends a formatted error string and moves on to the next file.

## 3. Chunking: `RecursiveChunker`

Located in [`src/rag/chunking/recursive_chunker.py`](../../src/rag/chunking/recursive_chunker.py).
Constructed with three parameters (defaults shown):

```python
RecursiveChunker(chunk_size=900, chunk_overlap=120, minimum_chunk_size=80)
```

- `chunk_size` — the target maximum characters per chunk.
- `chunk_overlap` — how many trailing characters of one chunk get repeated at the start of the
  next, so a sentence that would otherwise be cut at a chunk boundary still appears in full in at
  least one chunk.
- `minimum_chunk_size` — chunks shorter than this get merged into the previous chunk rather than
  standing alone (avoids tiny, low-information chunks).

Chunking happens in two passes:

**Pass 1 — split into sections by heading.** `_split_sections()` walks the document line by line
and groups lines under whatever heading most recently preceded them (`_looks_like_heading()`
recognizes Markdown `#` headers, or a short, capitalized, non-sentence-like line — see the bug
story in section 4 below for exactly how this is defined and why it matters).

**Pass 2 — split each section into chunks by sentence.** `_split_text()` walks through a
section's text sentence by sentence (`_split_sentences()` splits on `.`/`!`/`?` followed by
whitespace), accumulating sentences into the current chunk until adding the next one would exceed
`chunk_size`. At that point it closes the current chunk, and starts the next one with the last
`chunk_overlap` characters of the previous chunk prepended (`_with_overlap()`) — that's the
repeated-text overlap. A single sentence longer than `chunk_size` all by itself gets hard-split by
character count (`_split_long_sentence()`) rather than left as one oversized chunk. Finally,
`_merge_tiny_chunks()` folds anything under `minimum_chunk_size` into its predecessor.

Each resulting piece becomes a `Chunk` (`src/rag/chunking/chunk.py`):

```python
class Chunk(BaseModel):
    chunk_id: str          # "{document_id}:{index}", e.g. "AI-RMF-1stdraft.pdf:42"
    document_id: str
    chunk_index: int
    text: str
    source: str
    document_type: str
    parent_section: str | None = None   # the heading this chunk fell under
    metadata: dict[str, Any] = Field(default_factory=dict)
```

`chunk_id` is positional — `"{document_id}:{index}"` — which matters later: [Chapter 7](
07-evaluation-framework.md) explains that the golden evaluation dataset's "correct answer" ids are
tied to this exact numbering, so they break if chunking parameters change.

## 4. A real bug: how table-of-contents pages used to shred retrieval quality

This is worth walking through in detail because it's a genuine example of how a subtle chunking
bug can silently wreck a RAG system's answer quality without ever throwing an error.

`_looks_like_heading()` recognizes a line as heading-like if it's short (under 90 characters),
doesn't end in a period, and matches a capitalized-title-like pattern. That's a reasonable
heuristic for real section headings like "3. Understanding and Addressing Risks." But it also
matches something else perfectly: **a table of contents entry.** A line like:

```
Attributes of the AI RMF 3
```

(heading text, then a page number, extracted from a PDF's TOC page) is short, capitalized, and has
no trailing period — it satisfies the "looks like a heading" test just as well as a real section
title does.

The original version of `_split_sections()` closed off the current section and started a new one
*every time* it saw a heading-like line, with no further check. On a PDF's table-of-contents page,
which is a long run of consecutive TOC-entry lines, every single one of those lines triggered its
own new section — and since each "section" was just that one line, it became its own standalone
chunk containing nothing but that exact TOC phrase.

The consequence: if a user asked "what are the attributes of the AI RMF?", the TOC chunk
containing literally the phrase *"Attributes of the AI RMF"* would often outrank the real section
that actually explains those attributes in prose — because the TOC chunk **is** that exact phrase,
an unbeatable match for keyword/lexical scoring, while the real content chunk merely discusses the
topic using different words around it. Retrieval would surface a bare, contentless page reference
instead of the actual answer.

**The fix** (`_has_body_content()`): a section is only closed and a new one started once the
*current* section already contains at least one line that does *not* look like a heading — i.e.,
real body text. A run of consecutive heading-like lines (a TOC, or repeated PDF running headers/
footers) now keeps accumulating into one pending section instead of each becoming its own
near-empty chunk.

```python
def _has_body_content(self, lines: list[str]) -> bool:
    return any(not self._looks_like_heading(line) for line in lines)
```

**Measured effect on the real sample PDF**
(`sample_documents/AI-RMF-1stdraft.pdf`): chunk count went from **209 chunks before the fix to 148
chunks after** — 61 fewer chunks, all of them junk one-line TOC/header fragments that used to
compete with real content for search ranking. Because chunk ids are positional, this also meant
[`evaluation/golden_dataset.json`](../../evaluation/golden_dataset.json)'s 20 hand-picked
`relevant_chunk_ids` all had to be rebuilt against the corrected chunk output (see
[Chapter 7](07-evaluation-framework.md)).

The lesson generalizes: a chunking heuristic that looks reasonable in isolation can interact badly
with a specific, common document shape (here, PDF-extracted tables of contents) in a way that only
shows up as *degraded search relevance*, not as a crash or a test failure — which is exactly why
this class of bug is dangerous, and why [Chapter 7](07-evaluation-framework.md)'s golden-dataset
evaluation exists: to catch relevance regressions like this with a number, not a hunch.

## 5. Tracing a real ingest call end to end

Putting sections 2 and 3 together, here's exactly what happens inside `RAGService.ingest()`
(`src/app/services/rag_service.py`) for one file path:

```python
for file_path in file_paths:
    if not self._is_path_allowed(file_path):
        errors.append(f"{file_path}: PATH_NOT_ALLOWED ...")
        continue

    document_result = self.ingestion_pipeline.ingest_file(file_path)   # parse + clean
    if not document_result.success:
        errors.append(...); continue

    chunk_result = self.chunker.chunk(document_result.data)            # section + sentence split
    if not chunk_result.success:
        errors.append(...); continue

    records = [
        (chunk, self.embedder.embed(chunk.text))    # Chapter 3
        for chunk in chunk_result.data
    ]
    self.vector_store.add_many(records)              # Chapter 3
    indexed_documents += 1
    indexed_chunks += len(records)
```

Note the `_is_path_allowed()` check right at the top, before any file is even opened — this is a
security control covered in full in [Chapter 13](13-security-and-glossary.md): `/ingest` takes
attacker-controlled paths over an unauthenticated HTTP endpoint, so the live app restricts it to a
configured directory (`INGEST_ALLOWED_DIR`, default `sample_documents`) before touching the
filesystem at all. Direct construction of `RAGService()` (tests, `main.py`'s demo run) leaves this
unrestricted by default.

Next: [Chapter 3 — Embeddings & Vector Search](03-embeddings-and-vector-search.md).

import re
from src.models import Chunk, ChunkingStrategy
from src.extraction import PageSegment


STRATEGY_MAPPING = {
    "pdf": ChunkingStrategy.RECURSIVE,
    "docx": ChunkingStrategy.HEADING,
    "md": ChunkingStrategy.HEADING,
    "html": ChunkingStrategy.HEADING,
    "txt": ChunkingStrategy.RECURSIVE,
    "xlsx": ChunkingStrategy.TABLE,
    "csv": ChunkingStrategy.TABLE,
}


class SmartSplitter:
    """Intelligent text splitter that auto-selects strategy based on file type."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def _estimate_token_count(text: str) -> int:
        """Rough token estimate for storage/display only; char count keeps it dependency-free."""
        return len(text)

    def split_segments(
        self,
        segments: list[PageSegment],
        file_type: str = "txt",
        strategy: ChunkingStrategy = ChunkingStrategy.AUTO,
    ) -> list[Chunk]:
        """Position-aware split: each PageSegment is split independently so that
        every chunk carries page/line metadata for source traceability.

        For non-positional strategies (table, qa_pair, parent_child) the segments
        are concatenated and split as a whole — page metadata is set to the first
        segment's page number since these strategies rely on document-level structure.
        """
        if strategy == ChunkingStrategy.AUTO:
            strategy = STRATEGY_MAPPING.get(file_type, ChunkingStrategy.RECURSIVE)

        # Strategies that need full-document context: concatenate first
        if strategy in (ChunkingStrategy.TABLE, ChunkingStrategy.QA_PAIR,
                        ChunkingStrategy.PARENT_CHILD, ChunkingStrategy.HEADING):
            full_text = "\n\n".join(s["text"] for s in segments)
            chunks = self.split(full_text, file_type, strategy)
            # Attach page from first segment as approximate source
            first_page = segments[0]["page"] if segments else 1
            for ch in chunks:
                ch.metadata.setdefault("page", first_page)
            return chunks

        # Recursive / fixed / semantic: split per page to preserve line positions
        all_chunks: list[Chunk] = []
        for seg in segments:
            page_chunks = self._split_page_segment(seg, strategy)
            all_chunks.extend(page_chunks)

        # Re-index chunk ids sequentially across all pages
        for i, ch in enumerate(all_chunks):
            ch.id = f"chunk-{i:04d}"
            ch.index = i
        return all_chunks

    def _split_page_segment(self, seg: PageSegment, strategy: ChunkingStrategy) -> list[Chunk]:
        """Split one page segment and attach page + line_start + line_end to each chunk."""
        lines = seg["lines"]
        page = seg["page"]

        if not lines:
            return []

        chunks: list[Chunk] = []

        if strategy == ChunkingStrategy.FIXED:
            raw_chunks = self._split_fixed(seg["text"])
        else:
            raw_chunks = self._split_recursive(seg["text"])

        # Map each chunk back to its line range within the page
        # by scanning lines sequentially and matching chunk text
        line_cursor = 0
        for ch in raw_chunks:
            chunk_lines = [l for l in ch.text.splitlines() if l.strip()]
            start_line = line_cursor + 1  # 1-based
            # Advance cursor by the number of lines consumed
            consumed = len(chunk_lines)
            end_line = line_cursor + consumed
            line_cursor = end_line

            ch.metadata = {
                **ch.metadata,
                "page": page,
                "line_start": start_line,
                "line_end": min(end_line, len(lines)),
                "total_lines": len(lines),
            }
            chunks.append(ch)

        return chunks

    def split(
        self, text: str, file_type: str = "txt", strategy: ChunkingStrategy = ChunkingStrategy.AUTO
    ) -> list[Chunk]:
        if strategy == ChunkingStrategy.AUTO:
            strategy = STRATEGY_MAPPING.get(file_type, ChunkingStrategy.RECURSIVE)

        if strategy == ChunkingStrategy.HEADING:
            return self._split_by_heading(text)
        elif strategy == ChunkingStrategy.RECURSIVE:
            return self._split_recursive(text)
        elif strategy == ChunkingStrategy.FIXED:
            return self._split_fixed(text)
        elif strategy == ChunkingStrategy.QA_PAIR:
            return self._split_qa(text)
        elif strategy == ChunkingStrategy.PARENT_CHILD:
            return self._split_parent_child(text)
        elif strategy == ChunkingStrategy.TABLE:
            return self._split_table(text)
        elif strategy == ChunkingStrategy.SEMANTIC:
            return self._split_recursive(text)
        else:
            return self._split_recursive(text)

    def _split_by_heading(self, text: str) -> list[Chunk]:
        """Split by markdown headings or structural headers."""
        sections = re.split(r'(?:^|\n)(#{1,3}\s+.+)', text)
        chunks = []
        current_header = ""
        current_text = ""

        for section in sections:
            if re.match(r'^#{1,3}\s+', section):
                if current_text.strip():
                    chunks.append(self._make_chunk(
                        current_text.strip(), len(chunks), current_header
                    ))
                current_header = section.strip()
                current_text = ""
            else:
                current_text += section

        if current_text.strip():
            chunks.append(self._make_chunk(current_text.strip(), len(chunks), current_header))

        if not chunks and text.strip():
            return self._split_recursive(text)

        return chunks

    def _split_recursive(self, text: str) -> list[Chunk]:
        """Split by separators recursively: \\n\\n -> \\n -> sentence -> char."""
        separators = ["\n\n", "\n", "。", ".", " "]
        return self._recursive_split(text, separators)

    def _recursive_split(self, text: str, separators: list[str]) -> list[Chunk]:
        if len(text) <= self.chunk_size:
            if text.strip():
                return [self._make_chunk(text.strip(), 0)]
            return []

        sep = separators[0] if separators else ""
        remaining_seps = separators[1:] if len(separators) > 1 else []

        if sep:
            parts = text.split(sep)
        else:
            parts = [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        chunks = []
        current = ""

        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) > self.chunk_size and current:
                if len(current) > self.chunk_size and remaining_seps:
                    chunks.extend(self._recursive_split(current, remaining_seps))
                else:
                    chunks.append(self._make_chunk(current.strip(), len(chunks)))
                current = part
            else:
                current = candidate

        if current.strip():
            if len(current) > self.chunk_size and remaining_seps:
                chunks.extend(self._recursive_split(current, remaining_seps))
            else:
                chunks.append(self._make_chunk(current.strip(), len(chunks)))

        return chunks

    def _split_fixed(self, text: str) -> list[Chunk]:
        """Fixed-size splitting."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(self._make_chunk(chunk_text, len(chunks)))
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        return chunks

    def _split_qa(self, text: str) -> list[Chunk]:
        """Split by Q&A pairs."""
        pairs = re.split(r'\n(?=Q[:：]|问[:：])', text)
        chunks = []
        for pair in pairs:
            if pair.strip():
                chunks.append(self._make_chunk(pair.strip(), len(chunks)))
        if not chunks and text.strip():
            return self._split_recursive(text)
        return chunks

    def _split_table(self, text: str) -> list[Chunk]:
        """Split CSV/TSV into one chunk per data row, each prefixed with the header.

        Header line = first non-empty row. Every data row becomes its own chunk so
        retrieval returns exactly one record rather than a mixed multi-row blob.
        When a single row already exceeds chunk_size (e.g. very wide tables), it is
        still emitted as one chunk — splitting mid-row would destroy the record.
        """
        rows = [r for r in text.splitlines() if r.strip()]
        if not rows:
            return []

        header = rows[0]
        data_rows = rows[1:]

        if not data_rows:
            # Header-only file — return as a single chunk
            return [self._make_chunk(header, 0)]

        chunks = []
        for row in data_rows:
            chunk_text = f"{header}\n{row}"
            chunks.append(self._make_chunk(chunk_text, len(chunks)))
        return chunks

    def _split_parent_child(self, text: str) -> list[Chunk]:
        """Index child chunks while preserving parent section text for recall context."""
        sections = self._heading_sections(text)
        if not sections:
            sections = [("", text)]

        chunks: list[Chunk] = []
        for parent_index, (header, body) in enumerate(sections):
            parent_text = f"{header}\n\n{body}".strip() if header else body.strip()
            if not parent_text:
                continue
            child_source = body.strip() or parent_text
            children = self._split_recursive(child_source)
            for child_index, child in enumerate(children):
                chunks.append(self._make_chunk(
                    child.text,
                    len(chunks),
                    header,
                    metadata={
                        "parent_id": f"parent-{parent_index:04d}",
                        "parent_index": parent_index,
                        "child_index": child_index,
                        "parent_text": parent_text,
                    },
                ))
        return chunks

    def _heading_sections(self, text: str) -> list[tuple[str, str]]:
        sections = re.split(r'(?:^|\n)(#{1,3}\s+.+)', text)
        out: list[tuple[str, str]] = []
        current_header = ""
        current_text = ""
        for section in sections:
            if re.match(r'^#{1,3}\s+', section):
                if current_text.strip():
                    out.append((current_header, current_text.strip()))
                current_header = section.strip()
                current_text = ""
            else:
                current_text += section
        if current_text.strip():
            out.append((current_header, current_text.strip()))
        return out

    def _make_chunk(self, text: str, index: int, header: str = "", metadata: dict | None = None) -> Chunk:
        return Chunk(
            id=f"chunk-{index:04d}",
            doc_id="",
            kb_id="",
            text=text,
            index=index,
            metadata=metadata or {},
            context_header=header,
            token_count=self._estimate_token_count(text),
        )

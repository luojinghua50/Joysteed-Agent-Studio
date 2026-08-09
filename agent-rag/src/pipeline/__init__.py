import re
from src.models import Chunk, ChunkingStrategy


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

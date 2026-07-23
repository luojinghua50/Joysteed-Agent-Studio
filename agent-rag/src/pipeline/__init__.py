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

    def _make_chunk(self, text: str, index: int, header: str = "") -> Chunk:
        return Chunk(
            id=f"chunk-{index:04d}",
            doc_id="",
            kb_id="",
            text=text,
            index=index,
            context_header=header,
            token_count=len(text),
        )

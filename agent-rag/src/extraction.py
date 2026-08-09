"""Binary document text extraction.

Supports PDF (via pypdf) and Word docx (via python-docx).
Both libraries are optional — if not installed the function falls back to
UTF-8 decode with errors='ignore', which is the original behaviour for
plain-text formats.
"""
import io


def extract_text(content: bytes, file_type: str) -> str:
    """Return plain text from *content* bytes.

    file_type should be the lowercase extension without the leading dot,
    e.g. 'pdf', 'docx', 'txt', 'md'.
    """
    ft = file_type.lower()

    if ft == "pdf":
        return _extract_pdf(content)

    if ft in ("docx", "doc"):
        return _extract_docx(content)

    if ft in ("xlsx", "xls"):
        return _extract_xlsx(content)

    # Plain-text formats: md, txt, csv, tsv, json, yaml, yml, log, etc.
    return content.decode("utf-8", errors="ignore")


def _extract_pdf(content: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return content.decode("utf-8", errors="ignore")

    reader = PdfReader(io.BytesIO(content))
    parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _extract_docx(content: bytes) -> str:
    try:
        from docx import Document
    except ImportError:
        return content.decode("utf-8", errors="ignore")

    doc = Document(io.BytesIO(content))
    parts = []
    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text)
    # Also extract text from tables
    for table in doc.tables:
        for row in table.rows:
            row_text = "\t".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                parts.append(row_text)
    return "\n".join(parts)


def _extract_xlsx(content: bytes) -> str:
    """Convert xlsx to TSV-like text so the table splitter can handle it."""
    try:
        import openpyxl
    except ImportError:
        return content.decode("utf-8", errors="ignore")

    wb = openpyxl.load_workbook(io.BytesIO(content), read_only=True, data_only=True)
    parts = []
    for sheet in wb.worksheets:
        for row in sheet.iter_rows(values_only=True):
            row_text = "\t".join(str(v) for v in row if v is not None)
            if row_text.strip():
                parts.append(row_text)
    return "\n".join(parts)

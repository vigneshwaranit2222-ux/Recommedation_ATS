"""Robust PDF text extraction with pdfplumber.

The single public function `extract_text_from_pdf` reads a PDF file (or
in-memory bytes) and returns its concatenated text. It is defensive against
the common failure modes of resume PDFs:

  * corrupt / truncated files,
  * scanned (image-only) PDFs that yield no text,
  * password-protected PDFs,
  * generic pdfplumber/PDF syntax errors.

In every failure case we raise a `PDFParseError` with a human-readable
message so the FastAPI layer can map it to an appropriate HTTP response
without the worker process crashing.
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional, Union

import pdfplumber


class PDFParseError(Exception):
    """Raised when a PDF cannot be parsed or contains no extractable text.

    The message is safe to surface to the API consumer.
    """


# Path-like or bytes input. We accept both so the route can pass the
# UploadFile.file stream directly or a raw bytes blob.
PathOrBytes = Union[str, "os.PathLike[str]", bytes]


def _extract_from_path(path: str) -> str:
    """Open the PDF at `path` with pdfplumber and return all page text.

    Raises `PDFParseError` on any pdfplumber error or when the PDF appears to
    be scanned (no text extracted across all pages).
    """
    pages_text: list[str] = []
    try:
        # `pdfplumber.open` is a context manager that closes the file. We
        # iterate pages lazily to keep memory low for large PDFs.
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                # `page.extract_text()` can return None for image-only pages;
                # treat that as empty string and keep going.
                text = page.extract_text() or ""
                pages_text.append(text)
    except pdfplumber.pdfminer.pdfparser.PDFSyntaxError as exc:
        # Raised for corrupt / non-PDF / truncated files.
        raise PDFParseError(
            "The uploaded file is not a valid or readable PDF (corrupt or "
            "truncated). Please re-export the resume as a text-based PDF."
        ) from exc
    except Exception as exc:  # noqa: BLE001 - surface any other parse error
        # Catch-all for password-protected PDFs, encoding issues, etc.
        # We re-raise as our domain error so the API layer has one type to
        # handle.
        raise PDFParseError(f"Failed to parse PDF: {exc}") from exc

    full_text = "\n".join(pages_text).strip()

    # A scanned/image-only PDF will produce empty (or whitespace-only) text.
    # We treat this as a parse failure because downstream NER/scoring need
    # real text to work on.
    if not full_text:
        raise PDFParseError(
            "No selectable text could be extracted from this PDF. It appears "
            "to be a scanned/image-only resume. Please upload a text-based "
            "(digital) PDF."
        )

    return full_text


def extract_text_from_pdf(source: PathOrBytes) -> str:
    """Extract text from a PDF given a file path or raw bytes.

    Parameters
    ----------
    source:
        Either a filesystem path (str/PathLike) to a PDF, or the raw bytes
        of a PDF (e.g. from `await UploadFile.read()`).

    Returns
    -------
    str
        The concatenated, stripped text of all pages.

    Raises
    ------
    PDFParseError
        If the file is corrupt, password-protected, scanned/image-only, or
        otherwise unreadable.
    """
    # Bytes input: write to a temporary file so pdfplumber can open it by
    # path. We delete the temp file in the `finally` block regardless of
    # success/failure.
    if isinstance(source, (bytes, bytearray)):
        tmp_path: Optional[str] = None
        try:
            # `delete=False` because we need the path to exist for
            # pdfplumber.open; we clean up manually below.
            with tempfile.NamedTemporaryFile(
                suffix=".pdf", delete=False
            ) as tmp:
                tmp.write(source)
                tmp_path = tmp.name
            return _extract_from_path(tmp_path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    # Best-effort cleanup; don't mask the real error.
                    pass

    # Path input: ensure it exists, then extract.
    if isinstance(source, (str, os.PathLike)):
        path = os.fspath(source)
        if not os.path.exists(path):
            raise PDFParseError(f"PDF file not found: {path}")
        return _extract_from_path(path)

    # Unsupported input type.
    raise PDFParseError(
        f"Unsupported input type for PDF extraction: {type(source).__name__}"
    )
"""Helper utilities for extracting structured data from Carros na Web PDFs."""

from __future__ import annotations

import re
from pathlib import Path

import fitz
import pandas as pd


CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
WHITESPACE_RE = re.compile(r"[ \t]+")
NEWLINE_GAPS_RE = re.compile(r"\n{3,}")


def load_pdf(pdf_path: str | Path) -> fitz.Document:
    """Open a PDF document with PyMuPDF."""
    return fitz.open(str(pdf_path))


def clean_text(text: str) -> str:
    """Normalize extracted PDF text while preserving line breaks."""
    text = text.replace("\r", "\n")
    text = CONTROL_CHARS_RE.sub("", text)
    text = WHITESPACE_RE.sub(" ", text)
    text = NEWLINE_GAPS_RE.sub("\n\n", text)
    return text.strip()


def present(value: str = "Presente") -> str:
    """Convenience helper for feature rows where the PDF only states the feature name."""
    return value


def build_dataframe(rows: list[tuple[str, str]]) -> pd.DataFrame:
    """Build the final dataframe using the expected two-column schema."""
    return pd.DataFrame(rows, columns=["Atributo veicular", "Valor"])


def extract_page_text(pdf_path: str | Path, page_number: int) -> str:
    """Extract and clean the text from a 1-based page number."""
    doc = load_pdf(pdf_path)
    page = doc.load_page(page_number - 1)
    return clean_text(page.get_text("text"))


def iter_clean_page_lines(pdf_path: str | Path):
    """Yield cleaned non-empty lines from each page in reading order."""
    doc = load_pdf(pdf_path)
    for page_number in range(1, doc.page_count + 1):
        page = doc.load_page(page_number - 1)
        text = clean_text(page.get_text("text"))
        for line in text.splitlines():
            line = line.strip()
            if line:
                yield page_number, line

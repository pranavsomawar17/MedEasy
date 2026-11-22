"""
Article text extraction utilities for URLs, PDFs, and plain text.
"""

from __future__ import annotations

import io
import re
from typing import Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

try:
    import trafilatura
except ImportError:
    trafilatura = None


REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def _clean_whitespace(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    non_empty = [line for line in lines if line]
    return "\n\n".join(non_empty)


def extract_from_url(url: str, timeout: int = 15) -> Tuple[str, str]:
    """Fetch a health article URL and return (title, cleaned_body_text)."""
    url = url.strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Please enter a valid URL starting with http:// or https://")

    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not fetch URL: {exc}") from exc

    html = response.text
    soup = BeautifulSoup(html, "html.parser")

    # Extract title
    title = ""
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        title = h1.get_text(" ", strip=True)
    elif soup.title and soup.title.string:
        title = soup.title.string.strip()
    else:
        title = parsed.netloc

    # Primary extraction via trafilatura (best for removing boilerplate/nav/ads)
    body = ""
    if trafilatura is not None:
        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            favor_precision=True,
        )
        if extracted:
            body = _clean_whitespace(extracted)

    # Fallback extraction via BeautifulSoup
    if len(body) < 200:
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]):
            tag.decompose()

        container = soup.find("article") or soup.find("main") or soup.body or soup
        paragraphs = [
            p.get_text(" ", strip=True)
            for p in container.find_all(["p", "li", "h2", "h3"])
            if len(p.get_text(strip=True)) > 25
        ]
        body = _clean_whitespace("\n".join(paragraphs))

    if len(body) < 80:
        raise RuntimeError(
            "Could not extract enough readable article text from that link. "
            "The site may block automated requests or require login—try pasting the article text instead."
        )

    return title, body


def extract_from_pdf(file_bytes: bytes, filename: str) -> Tuple[str, str]:
    """Extract readable text from an uploaded PDF file."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(page_text.strip())

    body = _clean_whitespace("\n\n".join(pages))
    if len(body) < 40:
        raise RuntimeError("No selectable text found in this PDF (it may be a scanned image).")

    title = filename.rsplit(".", 1)[0]
    return title, body

"""SecondSelf — Web Page and HTML Content Extractor."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from adapters.base import BaseExtractor, ExtractionError, MalformedContentError
from adapters.models import ExtractedContent


class HTMLExtractor(BaseExtractor):
    """Adapter for web articles, URLs, and raw HTML markup.

    Uses trafilatura for high-quality boilerplate removal, falling back
    to BeautifulSoup and regex patterns. All heavy dependencies are lazy-loaded.

    Example:
        >>> extractor = HTMLExtractor()
        >>> content = extractor.extract("<h1>Title</h1><p>Clean article body.</p>")
        >>> content.title
        'Title'
        >>> 'Clean article body.' in content.text
        True
    """

    name = "html"
    supported_mimes = ("text/html", "application/xhtml+xml")
    supported_extensions = (".html", ".htm", ".xhtml")

    def can_handle(
        self,
        raw_input: str | bytes | Path,
        *,
        mime_type: str | None = None,
        filename: str | None = None,
    ) -> bool:
        if super().can_handle(raw_input, mime_type=mime_type, filename=filename):
            return True
        if isinstance(raw_input, str):
            s = raw_input.strip()
            # Detect URLs
            if s.startswith("http://") or s.startswith("https://"):
                return True
            # Detect HTML markup
            lower = s.lower()
            if "<html" in lower or "<!doctype" in lower or bool(re.search(r"<\w+[^>]*>", s)):
                return True
        return False

    def extract(
        self,
        raw_input: str | bytes | Path,
        *,
        source_uri: str | None = None,
        **kwargs: Any,
    ) -> ExtractedContent:
        origin_uri = source_uri
        html_content = ""
        http_status: int | None = None
        timeout = kwargs.get("timeout", 15.0)

        # 1. Fetch content from web URL if input is a URL
        if isinstance(raw_input, str) and (raw_input.startswith("http://") or raw_input.startswith("https://")):
            url = raw_input.strip()
            origin_uri = origin_uri or url
            parsed = urlparse(url)
            if not parsed.netloc:
                raise MalformedContentError(f"Malformed URL provided: {url}")

            headers = {
                "User-Agent": "SecondSelf-KnowledgeEngine/0.1.0 (https://github.com/secondself; mailto:contact@secondself.org)",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            }
            try:
                resp = requests.get(url, headers=headers, timeout=timeout)
                resp.raise_for_status()
                html_content = resp.text
                http_status = resp.status_code
            except requests.RequestException as exc:
                raise ExtractionError(f"Failed to fetch URL '{url}': {exc}") from exc

        elif isinstance(raw_input, Path):
            origin_uri = origin_uri or str(raw_input)
            html_content = raw_input.read_text(encoding="utf-8", errors="replace")
        elif isinstance(raw_input, bytes):
            html_content = raw_input.decode("utf-8", errors="replace")
        elif isinstance(raw_input, str):
            html_content = raw_input
        else:
            raise MalformedContentError(f"Unsupported HTML input type: {type(raw_input)}")

        if not html_content.strip():
            raise MalformedContentError("HTML content is empty.")

        # 2. Extract clean text using lazy-loaded trafilatura or BeautifulSoup
        extracted_text: str | None = None
        title = "Web Page"
        metadata: dict[str, Any] = {"format": "html"}
        warnings: list[str] = []

        if http_status is not None:
            metadata["http_status"] = http_status

        # Try Trafilatura
        try:
            import trafilatura
            extracted_text = trafilatura.extract(
                html_content,
                include_links=False,
                include_images=False,
                include_tables=True,
                no_fallback=False,
            )
            meta = trafilatura.extract_metadata(html_content)
            if meta:
                if meta.title:
                    title = meta.title
                if meta.author:
                    metadata["author"] = meta.author
                if meta.date:
                    metadata["date"] = meta.date
                if meta.sitename:
                    metadata["site_name"] = meta.sitename
        except ImportError:
            warnings.append("trafilatura not installed; falling back to BeautifulSoup.")
        except Exception as exc:
            warnings.append(f"trafilatura extraction warning: {exc}")

        # Fallback to BeautifulSoup
        if not extracted_text:
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html_content, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside"]):
                    tag.decompose()
                extracted_text = soup.get_text(separator="\n", strip=True)
                if soup.title and soup.title.string:
                    title = soup.title.string.strip()
            except ImportError:
                warnings.append("BeautifulSoup not installed; falling back to regex stripping.")

        # Final Fallback to Regex
        if not extracted_text:
            text = re.sub(r"<[^>]+>", " ", html_content)
            extracted_text = re.sub(r"\s+", " ", text).strip()
            warnings.append("Extracted text using primitive regex strip.")

        clean_text = extracted_text.strip() if extracted_text else ""
        if len(clean_text) < 10:
            raise MalformedContentError("Failed to extract meaningful text from HTML document.")

        return ExtractedContent(
            text=clean_text,
            title=title,
            source_type=self.name,
            source_uri=origin_uri,
            metadata=metadata,
            warnings=warnings,
        )

import asyncio
import re
from html.parser import HTMLParser
from typing import Dict, List, Optional, Union
from urllib.parse import urljoin, urlparse

import httpx

from hai.core import Document
from .base import BaseLoader


class HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.skip_tags = {"script", "style", "noscript", "iframe"}
        self.current_tag = None

    def handle_starttag(self, tag, _attrs):
        self.current_tag = tag
        if tag in {"p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6", "li"}:
            self.text_parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self.text_parts.append("\n")

    def handle_data(self, data):
        if self.current_tag not in self.skip_tags:
            text = data.strip()
            if text:
                self.text_parts.append(text + " ")

    def get_text(self) -> str:
        """Get extracted text with cleaned up whitespace."""
        text = "".join(self.text_parts)
        text = re.sub(r" +", " ", text)
        text = re.sub(r"\n\s*\n", "\n\n", text)
        return text.strip()


class WebLoader(BaseLoader):
    """
    Load content from web URLs.

    Supports:
    - Single URL or list of URLs
    - Automatic text extraction from HTML
    - Custom headers and timeout
    - Follow redirects
    - Retries on network errors and 5xx; 4xx fails immediately

    ```python
    loader = WebLoader(urls=["https://example.com"])
    loader = WebLoader("docs_loader", urls=["https://docs.python.org"], timeout=30)
    ```
    """

    def __init__(
        self,
        name: str = "WebLoader",
        *,
        urls: Union[str, List[str]],
        timeout: int = 30,
        headers: Optional[Dict[str, str]] = None,
        extract_links: bool = False,
        max_retries: int = 3,
        strict: bool = False,
    ):
        """
        Args:
            name: Node name
            urls: URL or list of URLs to fetch
            timeout: Request timeout in seconds (default: 30)
            headers: Custom HTTP headers
            extract_links: Extract links from page (default: False)
            max_retries: Maximum number of attempts per URL (default: 3)
            strict: Raise on the first URL that fails instead of skipping it
        """
        super().__init__(name, strict=strict)
        self.urls = urls if isinstance(urls, list) else [urls]
        self.timeout = timeout
        self.headers = headers or {"User-Agent": "hai-web-loader/0.1.0 (Python httpx)"}
        self.extract_links = extract_links
        self.max_retries = max_retries

    def _extract_text_from_html(self, html: str) -> str:
        extractor = HTMLTextExtractor()
        extractor.feed(html)
        return extractor.get_text()

    def _extract_links_from_html(self, html: str, base_url: str) -> List[str]:
        links = re.findall(r'href=["\']([^"\']+)["\']', html)
        absolute_links = [urljoin(base_url, link) for link in links]
        return absolute_links

    async def _fetch_url(self, client: httpx.AsyncClient, url: str) -> Document:
        attempts = max(1, self.max_retries)
        for attempt in range(attempts):
            try:
                response = await client.get(url)
                response.raise_for_status()

                html_content = response.text
                text_content = self._extract_text_from_html(html_content)

                metadata = {
                    "source": url,
                    "status_code": response.status_code,
                    "content_type": response.headers.get("content-type", ""),
                    "final_url": str(response.url),
                }

                if self.extract_links:
                    links = self._extract_links_from_html(html_content, url)
                    metadata["links"] = links
                    metadata["link_count"] = len(links)

                parsed_url = urlparse(url)
                name = f"{parsed_url.netloc}{parsed_url.path}"

                return Document(content=text_content, metadata=metadata, name=name)

            except (httpx.NetworkError, httpx.TimeoutException, httpx.HTTPStatusError) as e:
                client_error = isinstance(e, httpx.HTTPStatusError) and e.response.status_code < 500
                if client_error or attempt == attempts - 1:
                    raise
                await asyncio.sleep(0.5 * (2**attempt))
        raise AssertionError("unreachable")

    async def load(self) -> List[Document]:
        async with httpx.AsyncClient(
            timeout=self.timeout,
            headers=self.headers,
            follow_redirects=True,
            max_redirects=10,
        ) as client:
            return await self._load_each(self.urls, lambda url: self._fetch_url(client, url))

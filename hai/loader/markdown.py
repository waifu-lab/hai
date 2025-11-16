import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from hai.core import Document
from .text import TextLoader


class MarkdownLoader(TextLoader):
    """
    Optimized Markdown loader with advanced features.

    Features:
    - Extract plain text or keep markdown formatting
    - Parse YAML/TOML frontmatter
    - Extract structured metadata (headers, links, images, tables)
    - Pre-compiled regex patterns for better performance

    Examples:
    ```python
    # Keep markdown formatting
    loader = MarkdownLoader("md_loader", source="./README.md")

    # Extract plain text only
    loader = MarkdownLoader("md_loader", source="./docs", extract_text=True)

    # Extract metadata
    loader = MarkdownLoader(
        "md_loader",
        source="./docs",
        extract_metadata=True
    )
    ```
    """

    _PATTERNS = {
        # Frontmatter
        "yaml_frontmatter": re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL),
        "toml_frontmatter": re.compile(r"^\+\+\+\s*\n(.*?)\n\+\+\+\s*\n", re.DOTALL),
        # Code blocks
        "code_block_fenced": re.compile(r"```[\s\S]*?```", re.MULTILINE),
        "code_block_indented": re.compile(r"^(?:    |\t).*$", re.MULTILINE),
        "code_inline": re.compile(r"`[^`\n]+`"),
        # Links and images
        "image": re.compile(r"!\[([^\]]*)\]\(([^\)]+)\)"),
        "image_ref": re.compile(r"!\[([^\]]*)\]\[([^\]]*)\]"),
        "link": re.compile(r"\[([^\]]+)\]\(([^\)]+)\)"),
        "link_ref": re.compile(r"\[([^\]]+)\]\[([^\]]*)\]"),
        "autolink": re.compile(r"<(https?://[^>]+)>"),
        # Headers
        "header": re.compile(r"^(#{1,6})\s+(.+?)(?:\s+#*)?$", re.MULTILINE),
        # Horizontal rules (must be checked before emphasis to avoid conflicts)
        "hr": re.compile(r"^[ \t]*(-{3,}|\*{3,}|_{3,})[ \t]*$", re.MULTILINE),
        # Emphasis - improved patterns with better matching
        "bold_asterisk": re.compile(r"\*\*(.+?)\*\*"),
        "bold_underscore": re.compile(r"__(.+?)__"),
        "italic_asterisk": re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)"),
        "italic_underscore": re.compile(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)"),
        "strikethrough": re.compile(r"~~(.+?)~~"),
        # Other elements
        "blockquote": re.compile(r"^>\s*", re.MULTILINE),
        "list_unordered": re.compile(r"^\s*[-*+]\s+", re.MULTILINE),
        "list_ordered": re.compile(r"^\s*\d+\.\s+", re.MULTILINE),
        "table": re.compile(r"^\|(.+)\|\s*$", re.MULTILINE),
        "footnote": re.compile(r"\[\^([^\]]+)\]"),
        "html_tag": re.compile(r"<[^>]+>"),
        # Cleanup
        "multiple_newlines": re.compile(r"\n{3,}"),
        "whitespace": re.compile(r"[ \t]+"),
    }

    def __init__(
        self,
        name: str,
        source: Union[str, Path],
        pattern: str = "*.md",
        encoding: Optional[str] = None,
        recursive: bool = False,
        extract_text: bool = False,
        extract_frontmatter: bool = True,
        extract_metadata: bool = False,
        preserve_code_blocks: bool = False,
    ):
        """
        Args:
            name: Node name
            source: File path or directory path
            pattern: Glob pattern (*.md)
            encoding: Force specific encoding
            recursive: Search subdirectories
            extract_text: Convert markdown to plain text
            extract_frontmatter: Parse YAML/TOML frontmatter
            extract_metadata: Extract structured metadata like headers, links
            preserve_code_blocks: Keep code blocks when extracting metadata
        """
        super().__init__(
            name=name,
            source=source,
            pattern=pattern,
            encoding=encoding,
            recursive=recursive,
        )
        self.extract_text = extract_text
        self.extract_frontmatter = extract_frontmatter
        self.extract_metadata = extract_metadata
        self.preserve_code_blocks = preserve_code_blocks

    def _parse_frontmatter(self, content: str) -> tuple[dict, str]:
        """Extract YAML/TOML frontmatter from markdown."""
        frontmatter = {}

        yaml_match = self._PATTERNS["yaml_frontmatter"].match(content)
        if yaml_match:
            fm_text = yaml_match.group(1)
            frontmatter = self._parse_yaml_simple(fm_text)
            content = content[yaml_match.end() :]
            return frontmatter, content

        toml_match = self._PATTERNS["toml_frontmatter"].match(content)
        if toml_match:
            fm_text = toml_match.group(1)
            frontmatter = self._parse_toml_simple(fm_text)
            content = content[toml_match.end() :]
            return frontmatter, content

        return frontmatter, content

    def _parse_yaml_simple(self, yaml_text: str) -> dict:
        """Simple YAML parser for common frontmatter patterns."""
        result = {}
        current_key = None
        current_list = []

        for line in yaml_text.split("\n"):
            line = line.rstrip()
            if not line:
                continue

            if ":" in line and not line.startswith(" "):
                if current_key and current_list:
                    result[current_key] = current_list
                    current_list = []

                key, value = line.split(":", 1)
                current_key = key.strip()
                value = value.strip()

                if value:
                    result[current_key] = value.strip('"').strip("'")
                    current_key = None
            elif current_key and line.startswith(" "):
                stripped = line.strip()
                if stripped.startswith("- "):
                    current_list.append(stripped[2:].strip('"').strip("'"))
                else:
                    current_list.append(stripped.strip('"').strip("'"))

        # Save last list if exists
        if current_key and current_list:
            result[current_key] = current_list

        return result

    def _parse_toml_simple(self, toml_text: str) -> dict:
        """Simple TOML parser for common frontmatter patterns."""
        result = {}
        for line in toml_text.split("\n"):
            if "=" in line:
                key, value = line.split("=", 1)
                result[key.strip()] = value.strip().strip('"').strip("'")
        return result

    def _extract_metadata(self, content: str) -> Dict[str, Any]:
        """Extract structured metadata from markdown content."""
        metadata: Dict[str, Any] = {}

        headers = []
        for match in self._PATTERNS["header"].finditer(content):
            level = len(match.group(1))
            text = match.group(2).strip()
            headers.append({"level": level, "text": text})
        if headers:
            metadata["headers"] = headers

        links = []
        for match in self._PATTERNS["link"].finditer(content):
            links.append({"text": match.group(1), "url": match.group(2)})
        if links:
            metadata["links"] = links

        images = []
        for match in self._PATTERNS["image"].finditer(content):
            images.append({"alt": match.group(1), "url": match.group(2)})
        if images:
            metadata["images"] = images

        code_blocks = []
        for match in self._PATTERNS["code_block_fenced"].finditer(content):
            block = match.group(0)

            first_line = block.split("\n")[0]
            lang = first_line.replace("```", "").strip() or "text"
            code = "\n".join(block.split("\n")[1:-1])
            code_blocks.append({"language": lang, "code": code})
        if code_blocks:
            metadata["code_blocks"] = code_blocks

        table_lines = []
        in_table = False
        for line in content.split("\n"):
            if self._PATTERNS["table"].match(line):
                if not in_table:
                    table_lines.append([])
                    in_table = True
                table_lines[-1].append(line)
            else:
                in_table = False
        if table_lines:
            metadata["tables"] = ["\n".join(t) for t in table_lines]

        return metadata

    def _markdown_to_text(self, markdown: str) -> str:
        text = markdown

        if not self.preserve_code_blocks:
            text = self._PATTERNS["code_block_fenced"].sub("", text)
            text = self._PATTERNS["code_block_indented"].sub("", text)
        text = self._PATTERNS["code_inline"].sub("", text)

        text = self._PATTERNS["hr"].sub("", text)

        # ![alt](url) -> alt
        text = self._PATTERNS["image"].sub(r"\1", text)
        text = self._PATTERNS["image_ref"].sub(r"\1", text)

        # [text](url) -> text
        text = self._PATTERNS["link"].sub(r"\1", text)
        text = self._PATTERNS["link_ref"].sub(r"\1", text)
        text = self._PATTERNS["autolink"].sub(r"\1", text)

        text = self._PATTERNS["html_tag"].sub("", text)

        text = self._PATTERNS["header"].sub(r"\2", text)

        text = self._PATTERNS["strikethrough"].sub(r"\1", text)
        text = self._PATTERNS["bold_asterisk"].sub(r"\1", text)
        text = self._PATTERNS["bold_underscore"].sub(r"\1", text)
        text = self._PATTERNS["italic_asterisk"].sub(r"\1", text)
        text = self._PATTERNS["italic_underscore"].sub(r"\1", text)

        text = self._PATTERNS["blockquote"].sub("", text)

        text = self._PATTERNS["list_unordered"].sub("", text)
        text = self._PATTERNS["list_ordered"].sub("", text)

        text = self._PATTERNS["footnote"].sub("", text)

        text = self._PATTERNS["multiple_newlines"].sub("\n\n", text)
        text = self._PATTERNS["whitespace"].sub(" ", text)

        return text.strip()

    async def load(self) -> List[Document]:
        """Load markdown file(s) with optimized processing."""
        documents = await super().load()

        processed_docs = []
        for doc in documents:
            content = doc.content
            metadata = doc.metadata.copy()

            if self.extract_frontmatter:
                frontmatter, content = self._parse_frontmatter(content)
                metadata.update(frontmatter)

            if self.extract_metadata:
                extracted = self._extract_metadata(content)
                metadata.update(extracted)

            if self.extract_text:
                content = self._markdown_to_text(content)

            processed_doc = Document(content=content, metadata=metadata, name=doc.name)
            processed_docs.append(processed_doc)

        return processed_docs

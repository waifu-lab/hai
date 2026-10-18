import re
from typing import Any, Dict

from ftfy import fix_text

from hai.core import BaseNode

DOUBLE_QUOTE = re.compile(f"[{re.escape(r'«‹»›„‟❝❞❮❯〝〞〟＂')}]")
SINGLE_QUOTE = re.compile(f"[{re.escape(r'‛❛❜`´')}']")


class Cleaner(BaseNode):
    """Fix mojibake and broken Unicode, optionally normalising quote characters.

    Works both on a single ``text`` and on the ``documents`` a loader produced,
    so it can sit right after a loader: ``loader >> cleaner >> chunker``.

    Context:
        reads: ``text`` (str, optional), ``documents`` (List[Document], optional)
        writes: ``text`` and each document's ``content`` (cleaned in place)
    """

    def __init__(self, name: str = "Cleaner", fix_quotes: bool = True):
        super().__init__(name)
        self.fix_quotes = fix_quotes

    async def exec(self, context: Dict[str, Any]) -> Dict[str, Any]:
        text = context.get("text", "")
        if text:
            context["text"] = self.clean(text)

        for doc in context.get("documents", []):
            if doc.content:
                doc.content = self.clean(doc.content)

        return context

    def clean(self, text: str) -> str:
        text = fix_text(text)
        if self.fix_quotes:
            text = self._fix_quotes(text)
        return text

    @staticmethod
    def _fix_quotes(text: str) -> str:
        text = SINGLE_QUOTE.sub("'", text)
        text = DOUBLE_QUOTE.sub('"', text)
        return text

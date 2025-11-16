import re
from typing import Any, Dict

from hai.core import BaseNode
from ftfy import fix_text

DOUBLE_QUOTE = re.compile(f"[{re.escape(r'«‹»›„‟❝❞❮❯〝〞〟＂')}]")
SINGLE_QUOTE = re.compile(f"[{re.escape(r"'‛'❛❜`´''")}]")


class Cleaner(BaseNode):
    def __init__(self, name: str = "Cleaner", fix_quotes: bool = True):
        super().__init__(name)
        self.fix_quotes = fix_quotes

    async def exec(self, context: Dict[str, Any]) -> Dict[str, Any]:
        text = context.get("text", "")

        if text:
            text = fix_text(text)

            if self.fix_quotes:
                text = self._fix_quotes(text)

            context["text"] = text

        return context

    @staticmethod
    def _fix_quotes(text: str) -> str:
        text = SINGLE_QUOTE.sub("'", text)
        text = DOUBLE_QUOTE.sub('"', text)
        return text

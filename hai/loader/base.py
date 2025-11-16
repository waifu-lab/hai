from abc import abstractmethod
from typing import Any, Dict, List

from hai.core import BaseNode, Document


class BaseLoader(BaseNode):
    def __init__(self, name: str):
        super().__init__(name)
        self.documents: List[Document] = []

    @abstractmethod
    async def load(self) -> List[Document]:
        pass

    async def exec(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.documents = await self.load()
        context["documents"] = self.documents
        context["document_count"] = len(self.documents)
        return context

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union


class BaseNode(ABC):
    def __init__(self, name: str):
        self.name = name
        self._next_node: Optional["BaseNode"] = None
        self._branches: Dict[str, "BaseNode"] = {}

    def __rshift__(self, other: Union["BaseNode", Dict[str, "BaseNode"]]) -> "BaseNode":
        if isinstance(other, dict):
            self._branches = other
            return self
        elif isinstance(other, BaseNode):
            self._next_node = other
            return other
        else:
            raise TypeError(f"Unsupported type for >>: {type(other)}")

    @abstractmethod
    async def exec(self, context: Dict[str, Any]) -> Any:
        pass

    def next(self, result: Any) -> Optional["BaseNode"]:
        if isinstance(result, str) and result in self._branches:
            return self._branches[result]

        if isinstance(result, dict) and "next" in result:
            branch_key = result["next"]
            if branch_key in self._branches:
                return self._branches[branch_key]

        return self._next_node

    def exec_fallback(
        self, context: Dict[str, Any], exc: Exception
    ) -> Optional["BaseNode"]:
        raise exc

    async def execute(self, context: Dict[str, Any]) -> Optional["BaseNode"]:
        try:
            result = await self.exec(context)
        except Exception as e:
            return self.exec_fallback(context, e)

        return self.next(result)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

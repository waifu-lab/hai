from typing import Any, Dict, List, Optional

from httpx import AsyncClient

from .providers import ProviderEmbedding, ProviderInfo, ProviderText, resolve_api_key

_API_KEY_ENV = ("GEMINI_API_KEY", "GOOGLE_API_KEY")


class Google(ProviderInfo):
    def __init__(
        self,
        name: str = "Google",
        api_key: Optional[str] = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
    ):
        super().__init__(
            name=name,
            api_key=resolve_api_key(api_key, _API_KEY_ENV, "Google"),
            base_url=base_url,
        )

    async def models(self) -> List[str]:
        """Model ids without the ``models/`` prefix, e.g. ``gemini-2.5-flash``. Follows every result page."""
        names: List[str] = []
        params: Dict[str, Any] = {"pageSize": 1000}
        async with AsyncClient(timeout=30.0) as client:
            while True:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers={"x-goog-api-key": self.api_key},
                    params=params,
                )
                response.raise_for_status()
                res = response.json()
                names.extend(item["name"].removeprefix("models/") for item in res.get("models", []))
                page_token = res.get("nextPageToken")
                if not page_token:
                    return names
                params["pageToken"] = page_token


class GoogleText(ProviderText):
    def __init__(
        self,
        name: str = "Google",
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.7,
        max_tokens: int = 1500,
        stream: bool = False,
        max_retries: int = 3,
        timeout: float = 60.0,
        max_tool_rounds: int = 10,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
    ):
        super().__init__(
            name=name,
            api_key=resolve_api_key(api_key, _API_KEY_ENV, "Google"),
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            max_retries=max_retries,
            timeout=timeout,
            max_tool_rounds=max_tool_rounds,
        )


class GoogleEmbedding(ProviderEmbedding):
    def __init__(
        self,
        name: str = "GoogleEmbedding",
        api_key: Optional[str] = None,
        model: str = "text-embedding-004",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/embeddings",
        max_retries: int = 3,
        timeout: float = 30.0,
    ):
        super().__init__(
            name=name,
            api_key=resolve_api_key(api_key, _API_KEY_ENV, "Google"),
            model=model,
            base_url=base_url,
            max_retries=max_retries,
            timeout=timeout,
        )

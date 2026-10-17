from typing import List

from httpx import AsyncClient

from .providers import ProviderEmbedding, ProviderInfo, ProviderText


class Google(ProviderInfo):
    def __init__(
        self,
        name: str = "Google",
        api_key: str = "your_google_api_key",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
    ):
        super().__init__(
            name=name,
            api_key=api_key,
            base_url=base_url,
        )

    async def models(self) -> List[str]:
        async with AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30.0,
            )
            response.raise_for_status()
            res = response.json()
            return [item["baseModelId"] for item in res.get("data", [])]


class GoogleText(ProviderText):
    def __init__(
        self,
        name: str = "Google",
        api_key: str = "your_google_api_key",
        model: str = "gemini-2.5-flash",
        temperature: float = 0.7,
        max_tokens: int = 1500,
        stream: bool = False,
        max_retries: int = 3,
        timeout: float = 60.0,
    ):
        super().__init__(
            name=name,
            api_key=api_key,
            model=model,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            max_retries=max_retries,
            timeout=timeout,
        )


class GoogleEmbedding(ProviderEmbedding):
    def __init__(
        self,
        name: str = "GoogleEmbedding",
        api_key: str = "your_google_api_key",
        model: str = "text-embedding-004",
    ):
        super().__init__(
            name=name,
            api_key=api_key,
            model=model,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/embeddings",
        )

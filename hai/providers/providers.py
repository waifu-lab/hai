from typing import Any, Dict, Optional

from httpx import AsyncClient

from hai.core.node import BaseNode

# curl https://api.openai.com/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -H "Authorization: Bearer $OPENAI_API_KEY" \
#   -d '{
#     "model": "gpt-5",
#     "messages": [
#       {
#         "role": "developer",
#         "content": "You are a helpful assistant."
#       },
#       {
#         "role": "user",
#         "content": "Hello!"
#       }
#     ]
#   }'

# {
#   "id": "chatcmpl-B9MBs8CjcvOU2jLn4n570S5qMJKcT",
#   "object": "chat.completion",
#   "created": 1741569952,
#   "model": "gpt-4.1-2025-04-14",
#   "choices": [
#     {
#       "index": 0,
#       "message": {
#         "role": "assistant",
#         "content": "Hello! How can I assist you today?",
#         "refusal": null,
#         "annotations": []
#       },
#       "logprobs": null,
#       "finish_reason": "stop"
#     }
#   ],
#   "usage": {
#     "prompt_tokens": 19,
#     "completion_tokens": 10,
#     "total_tokens": 29,
#     "prompt_tokens_details": {
#       "cached_tokens": 0,
#       "audio_tokens": 0
#     },
#     "completion_tokens_details": {
#       "reasoning_tokens": 0,
#       "audio_tokens": 0,
#       "accepted_prediction_tokens": 0,
#       "rejected_prediction_tokens": 0
#     }
#   },
#   "service_tier": "default"
# }


class ProviderText(BaseNode):
    """
    OpenAI
    Anthropic
    Google
    ...
    """

    def __init__(
        self,
        name: str,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ):
        super().__init__(name)
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stream = stream
        self.base_url = base_url

    async def exec(self, context: Dict[str, Any]) -> Any:
        messages = context.get("messages", [])

        if not messages:
            raise ValueError("context 中必須包含 'messages'")

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": self.stream,
        }

        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with AsyncClient(timeout=60.0) as client:
            if self.stream:
                return await self._stream_request(client, headers, payload, context)
            else:
                return await self._request(client, headers, payload, context)

    async def _request(
        self,
        client: AsyncClient,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        context: Dict[str, Any],
    ) -> str:
        response = await client.post(self.base_url, headers=headers, json=payload)
        response.raise_for_status()

        data = response.json()
        content = data["choices"][0]["message"]["content"]

        context["openai_response"] = data
        context["assistant_message"] = content

        return content

    async def _stream_request(
        self,
        client: AsyncClient,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        context: Dict[str, Any],
    ) -> str:
        full_content = ""

        async with client.stream(
            "POST", self.base_url, headers=headers, json=payload
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]

                    if data_str == "[DONE]":
                        break

                    try:
                        import json

                        data = json.loads(data_str)
                        delta = data["choices"][0]["delta"]

                        if "content" in delta:
                            content_chunk = delta["content"]
                            full_content += content_chunk
                            print(content_chunk, end="", flush=True)

                    except json.JSONDecodeError:
                        continue

        print()
        context["assistant_message"] = full_content

        return full_content


class ProviderEmbedding(BaseNode):
    def __init__(
        self,
        name: str,
        api_key: str,
        base_url: str,
        model: str,
    ):
        super().__init__(name)
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    async def exec(self, context: Dict[str, Any]) -> Any:
        input_text = context.get("input")

        if not input_text:
            raise ValueError("context 中必須包含 'input'")

        payload = {
            "model": self.model,
            "input": input_text,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with AsyncClient(timeout=60.0) as client:
            response = await client.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()

            data = response.json()

            if isinstance(input_text, list):
                embeddings = [item["embedding"] for item in data["data"]]
                context["embeddings"] = embeddings
                return embeddings
            else:
                embedding = data["data"][0]["embedding"]
                context["embedding"] = embedding
                context["response"] = data
                return embedding

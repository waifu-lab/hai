from .providers import ProviderEmbedding, ProviderText


class GoogleText(ProviderText):
    def __init__(
        self,
        name: str = "Google",
        api_key: str = "your_google_api_key",
        model: str = "gemini-2.5-flash",
        temperature: float = 0.7,
        max_tokens: int = 1500,
    ):
        super().__init__(
            name=name,
            api_key=api_key,
            model=model,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
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

# if __name__ == "__main__":
#     google_provider = GoogleText(
#         api_key="",
#         model="gemini-2.5-flash",
#         temperature=0.7,
#         max_tokens=1500,
#     )
#     print(google_provider)
#     context = {
#         "messages": [
#             {"role": "system", "content": "你是一個有幫助的助手"},
#             {"role": "user", "content": "請用繁體中文介紹台灣的夜市文化, 200字以內"},
#         ]
#     }

#     import asyncio

#     async def run():
#         response = await google_provider.exec(context)
#         print(response)

#     asyncio.run(run())

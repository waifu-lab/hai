# Hydrogen AI (hai)

A lightweight async AI toolkit.

- No dependency on any official vendor SDK
- Text models: OpenAI, Anthropic, Google Gemini, Ollama, or any OpenAI-compatible service
- Speech: OpenAI STT / TTS, GPT-SoVITS
- Vector and graph databases as plugins: Qdrant, sqlite-vec, Neo4j

## Installation

```bash
pip install hydrogen-ai
```

Requires Python 3.9+. Database plugins are installed separately; see the [docs](docs/content/docs/plugins/index.mdx).

## Quick start

```python
import asyncio

from hai.core import Flow
from hai.providers import create_text_provider


async def main():
    # Reads OPENAI_API_KEY when api_key is not given
    llm = create_text_provider("openai", model="gpt-4o-mini")
    result = await Flow(llm).timeout(60).run({
        "messages": [{"role": "user", "content": "Explain vector search in one sentence."}]
    })
    print(result["assistant_message"])


asyncio.run(main())
```

## Documentation

The full documentation lives in [`docs/`](docs/content/docs) (written in Traditional Chinese).

## Development

```bash
poetry sync
poetry run pytest
poetry run ruff check hai tests
```

## Acknowledgements

The node and flow design is inspired by [PocketFlow](https://github.com/The-Pocket/PocketFlow). Thanks to the PocketFlow authors and contributors.

## License

MIT

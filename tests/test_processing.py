import asyncio

from hai.core import BaseNode, Document
from hai.embedders import EmbeddingProcessor
from hai.loader.web import HTMLTextExtractor
from hai.preprocessor import Cleaner
from hai.processor import Splitter


def extract(html):
    extractor = HTMLTextExtractor()
    extractor.feed(html)
    return extractor.get_text()


def test_html_keeps_text_after_script():
    text = extract("<div><script>bad()</script>visible tail</div><p>ok</p>")
    assert "visible tail" in text
    assert "bad()" not in text
    assert "ok" in text


def test_html_skips_nested_skip_tags():
    assert extract("<noscript><style>x{}</style>hidden</noscript>shown") == "shown"


def test_splitter_respects_chunk_size_and_overlap():
    splitter = Splitter(chunk_size=10, chunk_overlap=4, separator=" ", keep_separator=True)
    chunks = splitter._split_text("aa bb cc dd ee ff")
    assert all(len(c) <= 10 for c in chunks)
    assert "".join(chunks).replace(" ", "").startswith("aabbcc")
    assert len(chunks) > 1
    # The tail of one chunk starts the next
    assert chunks[1].startswith(chunks[0].split(" ")[-2])


def test_cleaner_cleans_loader_documents():
    context = {"documents": [Document("â€œhelloâ€\x9d", name="a.txt")]}
    asyncio.run(Cleaner().exec(context))
    assert context["documents"][0].content == '"hello"'


class FakeEmbedder(BaseNode):
    def __init__(self, drop=0):
        super().__init__("fake")
        self.drop = drop

    async def exec(self, context):
        texts = context["input"]
        context["embeddings"] = [[float(len(t))] for t in texts][: len(texts) - self.drop]


def test_embedding_processor_does_not_mutate_input():
    chunks = [{"content": "ab"}, {"content": "abc"}, {"content": "a"}]
    context = {"chunks": chunks}
    asyncio.run(EmbeddingProcessor(embedder=FakeEmbedder(), batch_size=2).exec(context))
    assert [c["embedding"] for c in context["embedded_chunks"]] == [[2.0], [3.0], [1.0]]
    assert "embedding" not in chunks[0]


def test_embedding_processor_rejects_short_response():
    context = {"chunks": [{"content": "a"}, {"content": "b"}]}
    try:
        asyncio.run(EmbeddingProcessor(embedder=FakeEmbedder(drop=1)).exec(context))
    except ValueError as e:
        assert "1 embeddings for 2 texts" in str(e)
    else:
        raise AssertionError("expected ValueError")

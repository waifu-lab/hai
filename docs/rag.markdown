# RAG 文件檢索與生成範例

> [!IMPORTANT]
> 之後記得刪除


```python
import asyncio
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.append(str(Path(__file__).resolve().parent.parent))

from hai.core.node.node import BaseNode
from hai.embedders.openai.openai import EmbeddingProcessor, QueryEmbedder
from hai.processor import ChunkProcessor, RecursiveSplitter
from hai.providers.openai import OpenAIEmbedding
from hai.retriever.qdrant.qdrantStoreRetriever import VectorStoreRetriever
from hai.vectorstores.qdrant.QdrantClient import QdrantStore
from hai.vectorstores.qdrant.writer import QdrantStoreWriter


class DocumentLoader(BaseNode):
    def __init__(
        self,
        name: str = "DocumentLoader",
    ):
        super().__init__(name)

    async def exec(self, context: Dict[str, Any]) -> Dict[str, Any]:
        documents = context.get("documents", [])

        if isinstance(documents, str):
            documents = [{"content": documents, "name": "document"}]
        elif (
            isinstance(documents, list) and documents and isinstance(documents[0], str)
        ):
            documents = [
                {"content": doc, "name": f"document_{i}"}
                for i, doc in enumerate(documents)
            ]

        context["documents"] = documents
        context["document_count"] = len(documents)

        return context


class ContextBuilder(BaseNode):
    def __init__(
        self,
        name: str = "ContextBuilder",
        template: str = "根據以下資料回答問題：\n\n{context}\n\n問題：{query}",
    ):
        super().__init__(name)
        self.template = template

    async def exec(self, context: Dict[str, Any]) -> Dict[str, Any]:
        query = context.get("query", "")
        documents = context.get("retrieved_documents", [])

        doc_texts = []
        for i, doc in enumerate(documents):
            doc_texts.append(f"[{i + 1}] {doc['content']}")

        context_text = "\n\n".join(doc_texts)

        prompt = self.template.format(context=context_text, query=query)
        context["prompt"] = prompt

        return context

async def test_retrieval_pipeline():
    print("\n=== 測試檢索管道 ===")

    store = QdrantStore(
        location=":memory:",
        collection_name="test_retrieval",
        embedding_dim=1536,
    )

    embedder = OpenAIEmbedding(
        name="embedder",
        api_key="",
        model="text-embedding-3-small",
    )

    query_embed = QueryEmbedder("query_embed", embedder=embedder)
    retriever = VectorStoreRetriever(
        "retriever", store=store, top_k=3, score_threshold=0.5
    )
    context_builder = ContextBuilder(
        "context_builder",
        template="請根據以下參考資料回答問題：\n\n{context}\n\n使用者問題：{query}\n\n請提供詳細的回答：",
    )

    query_embed >> retriever >> context_builder

    context = {"query": "文件中提到了哪些內容？"}

    current = query_embed
    while current:
        current = await current.execute(context)

    print(f"檢索到 {context['retrieved_count']} 個相關文件")
    print(f"\n生成的 Prompt:\n{context['prompt']}")


async def test_full_rag_pipeline():
    print("\n=== 測試完整 RAG 管道 ===")

    store = QdrantStore(
        location=":memory:",
        collection_name="full_rag",
        embedding_dim=1536,
    )

    embedder = OpenAIEmbedding(
        name="embedder",
        api_key="",
        model="text-embedding-3-small",
    )

    splitter = RecursiveSplitter(chunk_size=200, chunk_overlap=50)

    loader = DocumentLoader("loader")
    chunker = ChunkProcessor("chunker", splitter=splitter)
    embed_proc = EmbeddingProcessor("embed", embedder=embedder)
    writer = QdrantStoreWriter("writer", store=store)

    loader >> chunker >> embed_proc >> writer

    ingest_context = {
        "documents": [
            {
                "content": "台北101大樓是台灣最著名的地標建築，高度508公尺，共有101層樓。大樓內設有購物中心、觀景台和多家餐廳。觀景台位於89樓，可以俯瞰整個台北市區。每年跨年時，101大樓會施放壯觀的煙火秀，吸引數十萬人觀賞。",
                "name": "taipei101.txt",
            },
            {
                "content": "故宮博物院收藏了超過69萬件中華文化珍貴文物。鎮館之寶包括翠玉白菜、肉形石和毛公鼎。翠玉白菜是用翡翠雕刻而成，栩栩如生。博物院每年吸引數百萬遊客參觀，是台灣最重要的文化機構之一。",
                "name": "national_palace_museum.txt",
            },
            {
                "content": "日月潭是台灣最大的天然湖泊，位於南投縣。湖面海拔748公尺，因北半部形似日輪、南半部形似月鉤而得名。著名景點包括文武廟、玄光寺和九族文化村。遊客可以搭乘遊艇環湖，或騎自行車繞湖一周。",
                "name": "sun_moon_lake.txt",
            },
        ]
    }

    print("1. 導入文件...")
    current = loader
    while current:
        current = await current.execute(ingest_context)
    print(f"   文件數: {ingest_context['document_count']}")
    print(f"   Chunks: {ingest_context['total_chunks']}")
    print(f"   已存儲: {ingest_context['stored_count']} 個 chunks")

    # === 檢索與生成 Prompt ===

    query_embed = QueryEmbedder("query_embed", embedder=embedder)
    retriever = VectorStoreRetriever("retriever", store=store, top_k=2)
    context_builder = ContextBuilder("context_builder")

    query_embed >> retriever >> context_builder

    test_queries = [
        "台北101有多高？",
        "故宮博物院有什麼著名的文物？",
        "日月潭在哪裡？有什麼特色？",
    ]

    for query in test_queries:
        print(f"\n2. 查詢: {query}")

        query_context = {"query": query}
        current = query_embed
        while current:
            current = await query_context_exec(current, query_context)

        print(f"   找到 {query_context['retrieved_count']} 個相關文件")

        for i, doc in enumerate(query_context["retrieved_documents"]):
            print(f"   [{i + 1}] 分數: {doc['score']:.4f}")
            print(f"       來源: {doc['name']}")
            print(f"       內容: {doc['content'][:80]}...")

    print("\n3. 生成完整 Prompt 範例:")
    final_query = {"query": "台北有哪些著名景點？"}
    current = query_embed
    while current:
        current = await query_context_exec(current, final_query)

    print(final_query["prompt"])


async def query_context_exec(node, context):
    result = node.exec(context)
    if asyncio.iscoroutine(result):
        await result
    return node.next(None)


asyncio.run(test_full_rag_pipeline())

```
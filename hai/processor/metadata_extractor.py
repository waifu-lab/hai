from typing import Any, Dict, Optional

from hai.core import BaseNode


class MetadataExtractor(BaseNode):
    def __init__(
        self,
        name: str = "MetadataExtractor",
        llm: Optional[BaseNode] = None,
        extraction_type: str = "facts",
        system_prompt: Optional[str] = None,
    ):
        super().__init__(name)
        self.llm = llm
        self.extraction_type = extraction_type
        self.system_prompt = system_prompt or self._get_default_prompt()

    def _get_default_prompt(self) -> str:
        if self.extraction_type == "facts":
            return """You are a fact extraction expert.
Extract facts as triplets from the text.

Return a JSON object with this structure:
{
  "facts": [
    {
      "subject": "entity name",
      "predicate": "relationship type",
      "object": "related entity or value",
      "confidence": 0.9
    }
  ],
  "entities": [
    {
      "name": "entity name",
      "type": "Person|Organization|Location|Event|Product|Concept"
    }
  ]
}

Guidelines:
- Extract only factual statements, not opinions
- Use clear, normalized predicate names (e.g., "works_at", "located_in")
- Confidence: 0-1, where 1 is absolutely certain
- Include all mentioned entities with their types
"""
        elif self.extraction_type == "entities":
            return """Extract all entities from the text with their types.

Return a JSON object:
{
  "entities": [
    {
      "name": "entity name",
      "type": "Person|Organization|Location|Event|Product|Concept",
      "mentions": 1
    }
  ]
}
"""
        else:
            return "Extract structured information from the text as JSON."

    async def exec(self, context: Dict[str, Any]) -> Dict[str, Any]:
        chunks = context.get("chunks", [])

        if not chunks:
            context["extracted_metadata"] = []
            return context

        if not self.llm:
            raise ValueError("LLM is required for metadata extraction")

        all_metadata = []

        for chunk in chunks:
            content = chunk.get("content", "")
            if not content.strip():
                continue

            llm_context = {
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": content},
                ]
            }

            try:
                result = await self.llm.exec(llm_context)

                if "parsed_response" in llm_context:
                    metadata = llm_context["parsed_response"]
                elif isinstance(result, dict):
                    metadata = result
                else:
                    import json

                    metadata = json.loads(result)

                chunk_metadata = {
                    "chunk_id": chunk.get("chunk_index", 0),
                    "chunk_name": chunk.get("name", "unknown"),
                    "source": chunk.get("source", ""),
                    **metadata,
                }

                all_metadata.append(chunk_metadata)

            except Exception as e:
                print(f"Error extracting metadata from chunk: {e}")
                continue

        context["extracted_metadata"] = all_metadata
        context["metadata_count"] = len(all_metadata)

        return context


class FactExtractor(MetadataExtractor):
    def __init__(
        self,
        name: str = "FactExtractor",
        llm: Optional[BaseNode] = None,
        min_confidence: float = 0.7,
    ):
        super().__init__(
            name=name,
            llm=llm,
            extraction_type="facts",
        )
        self.min_confidence = min_confidence

    async def exec(self, context: Dict[str, Any]) -> Dict[str, Any]:
        await super().exec(context)

        extracted_metadata = context.get("extracted_metadata", [])
        all_facts = []
        all_entities = []

        for item in extracted_metadata:
            facts = item.get("facts", [])
            entities = item.get("entities", [])

            for fact in facts:
                if fact.get("confidence", 0) >= self.min_confidence:
                    fact_with_source = {
                        **fact,
                        "source": f"{item['chunk_name']}/chunk_{item['chunk_id']}",
                        "chunk_id": item["chunk_id"],
                    }
                    all_facts.append(fact_with_source)

            for entity in entities:
                entity_with_source = {
                    **entity,
                    "source": f"{item['chunk_name']}/chunk_{item['chunk_id']}",
                }
                all_entities.append(entity_with_source)

        context["extracted_facts"] = all_facts
        context["extracted_entities"] = all_entities
        context["fact_count"] = len(all_facts)
        context["entity_count"] = len(all_entities)

        return context


class EntityExtractor(MetadataExtractor):
    def __init__(
        self,
        name: str = "EntityExtractor",
        llm: Optional[BaseNode] = None,
    ):
        super().__init__(
            name=name,
            llm=llm,
            extraction_type="entities",
        )

    async def exec(self, context: Dict[str, Any]) -> Dict[str, Any]:
        await super().exec(context)

        extracted_metadata = context.get("extracted_metadata", [])
        all_entities = []
        entity_map = {}

        for item in extracted_metadata:
            entities = item.get("entities", [])

            for entity in entities:
                name = entity.get("name")
                if name in entity_map:
                    entity_map[name]["mentions"] += entity.get("mentions", 1)
                else:
                    entity_map[name] = {
                        **entity,
                        "mentions": entity.get("mentions", 1),
                        "sources": [],
                    }

                entity_map[name]["sources"].append(
                    f"{item['chunk_name']}/chunk_{item['chunk_id']}"
                )

        all_entities = list(entity_map.values())

        context["extracted_entities"] = all_entities
        context["entity_count"] = len(all_entities)

        return context

import logging
from typing import Any, Dict, Optional

from hai.core import BaseNode

logger = logging.getLogger(__name__)

_FACTS_SYSTEM_PROMPT = """\
You are a knowledge graph extraction expert.
Your job is to extract anchor entities and relationships from text, then return JSON.

Output schema (strict JSON, no extra keys):
{
  "entities": [
    {"name": "<string>", "type": "Person|Organization|Location|Substance|Product"}
  ],
  "facts": [
    {
      "subject": "<entity name>",
      "predicate": "<SCREAMING_SNAKE_CASE>",
      "object": "<entity name>",
      "fact": "<one natural-language sentence capturing full context>",
      "confidence": <0.0-1.0>,
      "valid_from": "<YYYY-MM-DD or null>",
      "valid_until": "<YYYY-MM-DD or null>"
    }
  ]
}
"""

_FACTS_USER_TEMPLATE = """\
<forbidden_entities>
The following must NEVER be extracted as entities, regardless of context:

- Pronouns — words that substitute for a noun without naming it
  <examples>he, she, it, they, him, her, them, his, hers, its, their, I, me</examples>

- Abstract concepts or feelings — intangible states with no real-world referent
  <examples>fair, good, growth, great, happy, joyful, quiet, serious</examples>

- Common nouns — generic object or role words with no unique identity
  <examples>pet, game, park, road, river, queen, city, book, key, dog, cat, doctor</examples>

- Generic media nouns — content types without a specific title or name
  <examples>movie, song, book, article, video, podcast, image</examples>

- Generic event nouns — activity types without a specific named instance
  <examples>party, meeting, concert, trip, vacation, workout, run, walk, flight</examples>

- Broad institutional nouns — category labels without an explicit proper name
  <examples>company, organization, hospital, school, university, government</examples>

- Temporal expressions — bare time words without a specific named date
  <examples>morning, tonight, yesterday, last week, next month, recently, soon, later</examples>

- Quantifiers or ordinals — vague amounts or positional references
  <examples>some, many, few, several, a lot, first, second, the rest, another, others</examples>

- Unnamed persons — Person entities MUST be proper nouns that uniquely identify
  one individual, not a role, title, or occupation.

  Ask: "Is this a name or a job?"
    name  → extract.   e.g. Wraith, Dr. Graves, Kelvin
    job   → FORBIDDEN. e.g. doctor, nurse, teacher, manager, engineer, lawyer,
                             officer, therapist, surgeon, clerk, driver, chef
  Possessor qualification does NOT promote a job title into a valid entity:
    "Wraith's doctor"  → FORBIDDEN (still a job, not a name)
    "his manager"      → FORBIDDEN
    "Ivy's dad"        → OK only for kinship terms with possessor, NOT for job titles
- Bare kinship terms — only extract when qualified with the possessor's name:
    dad, mom, sister, brother, husband, wife, son, daughter,
    uncle, aunt, cousin, grandma, grandpa, friend, neighbor, roommate
    → "Ivy's dad" YES, "dad" NO, "his brother" NO.
- Abstract conditions, symptoms, or states:
    insomnia, fatigue, health problems, immune decline, stress, pain, illness ...
</forbidden_entities>

<allowed_entities>
Only extract entities that are specifically and uniquely identifiable.

Allowed types:
  Person       — named individuals with a proper name (Wraith, Dr. Graves)
  Organization — named institutions, hospitals, companies (City Hospital, Google)
  Location     — named places, cities, addresses (New York, Central Park)
  Substance    — named chemicals, foods, drugs (caffeine, aspirin, coffee)
  Product      — named specific products or brands
</allowed_entities>

<relationship_rules>
- Relationships must connect two DISTINCT entities from your "entities" list.
- `fact` field: one sentence capturing the FULL context — causes, effects, frequency,
  dates, outcomes — for details NOT modelled as separate entity nodes.
- predicates in SCREAMING_SNAKE_CASE (LIKES_TO_CONSUME, CAUSES_HARM_TO, VISITS ...)
- valid_from / valid_until: ISO date if stated or inferable, otherwise null.
- Every entity MUST appear in at least one fact. No orphan entities.
- subject/object MUST exactly match a name in your "entities" list.
- Do NOT create possessive compound names (BAD: "Wraith's insomnia").
</relationship_rules>

<example>
Input:
  "Wraith loves drinking coffee every afternoon. The caffeine in coffee caused Wraith
   to suffer from chronic insomnia and fatigue. Wraith visited City Hospital in March
   2024. The doctor advised Wraith to stop drinking coffee."

Correct output:
  entities: Wraith (Person), coffee (Substance), caffeine (Substance),
            City Hospital (Organization)
  — doctor is NOT extracted: no proper name (job title)
  — insomnia, fatigue are NOT extracted: symptoms, not named entities
  — afternoon, March 2024 are NOT extracted: temporal expressions

Wrong output (do NOT do this):
  entities: Wraith (Person), coffee (Substance), caffeine (Substance),
            City Hospital (Organization),
            doctor (Person),        ← FORBIDDEN: job title, not a proper name
            insomnia (Condition),   ← FORBIDDEN: symptom / abstract state
            fatigue (Condition),    ← FORBIDDEN: symptom / abstract state
            afternoon (Time)        ← FORBIDDEN: temporal expression
</example>

<text_to_process>
{content}
</text_to_process>
"""


class MetadataExtractor(BaseNode):
    """Ask *llm* (a ProviderText) to extract structured JSON from every chunk.

    Context:
        reads: ``chunks`` (List[dict] from ChunkProcessor)
        writes: ``extracted_metadata`` (List[dict], one per chunk), ``metadata_count`` (int)
    """

    def __init__(
        self,
        name: str = "MetadataExtractor",
        llm: Optional[BaseNode] = None,
        extraction_type: str = "facts",
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
    ):
        """
        Args:
            system_prompt: Overrides the default system prompt of *extraction_type*.
            user_prompt_template: User message per chunk. Every ``{content}`` is replaced
                with the chunk text; other braces (e.g. JSON examples) are left as-is and
                need no escaping. Without a template the chunk text is sent unchanged
                (``"facts"`` has a built-in template).
        """
        super().__init__(name)
        self.llm = llm
        self.extraction_type = extraction_type

        if extraction_type == "facts":
            self.system_prompt = system_prompt or _FACTS_SYSTEM_PROMPT
            self.user_prompt_template = user_prompt_template or _FACTS_USER_TEMPLATE
        else:
            self.system_prompt = system_prompt or self._get_default_prompt()
            self.user_prompt_template = user_prompt_template

    def _get_default_prompt(self) -> str:
        if self.extraction_type == "entities":
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

            if self.user_prompt_template:
                user_content = self.user_prompt_template.replace("{content}", content)
            else:
                user_content = content

            llm_context = {
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_content},
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
                logger.error(f"Error extracting metadata from chunk: {e}")
                continue

        context["extracted_metadata"] = all_metadata
        context["metadata_count"] = len(all_metadata)

        return context


class FactExtractor(MetadataExtractor):
    """Extract knowledge-graph entities and facts, keeping facts at or above *min_confidence*.

    Context:
        reads: ``chunks`` (List[dict])
        writes: everything MetadataExtractor writes, plus ``extracted_facts`` (List[dict]),
            ``extracted_entities`` (List[dict]), ``fact_count`` (int), ``entity_count`` (int)
    """

    def __init__(
        self,
        name: str = "FactExtractor",
        llm: Optional[BaseNode] = None,
        min_confidence: float = 0.7,
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
    ):
        super().__init__(
            name=name,
            llm=llm,
            extraction_type="facts",
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
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
    """Extract entities and merge them by name across chunks, summing mentions.

    Context:
        reads: ``chunks`` (List[dict])
        writes: everything MetadataExtractor writes, plus ``extracted_entities`` (List[dict]),
            ``entity_count`` (int)
    """

    def __init__(
        self,
        name: str = "EntityExtractor",
        llm: Optional[BaseNode] = None,
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
    ):
        super().__init__(
            name=name,
            llm=llm,
            extraction_type="entities",
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
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

                entity_map[name]["sources"].append(f"{item['chunk_name']}/chunk_{item['chunk_id']}")

        all_entities = list(entity_map.values())

        context["extracted_entities"] = all_entities
        context["entity_count"] = len(all_entities)

        return context

"""Contract test: Schema detection invariant.

Validates that mem0ai's fact extraction messages contain a system message
and memory update messages do not. This is a structural invariant in
mem0ai's architecture — if it breaks, our schema detection fails.

NOTE (mem0ai 2.x): the presence of a system message is NECESSARY but no longer
SUFFICIENT to identify the legacy fact-extraction call. mem0ai 2.x replaced the
"Personal Information Organizer" prompt (which emitted {"facts": [...]}) with a
single-pass "Memory Extractor" prompt that ALSO carries a system message but
emits {"memory": [...]}. Selecting FACT_RETRIEVAL_SCHEMA purely on has_system
forced the wrong key and silently dropped every extraction. _select_schema()
is therefore content-aware; see TestV2PromptShift below.
"""

from __future__ import annotations

import pytest


class TestSchemaDetectionInvariant:
    """Contract test: verify the system message presence invariant.

    These tests validate the assumption that:
    - Fact extraction calls include a system message (prompt template)
    - Memory update calls use only a user message (no system message)

    This invariant is how we select FACT_RETRIEVAL_SCHEMA vs MEMORY_UPDATE_SCHEMA.
    """

    def test_fact_extraction_has_system_message(self):
        """Fact extraction prompt templates should have role=system."""
        # Simulate what mem0ai sends for fact extraction
        # The FACT_RETRIEVAL_PROMPT template is passed as system message
        fact_extraction_messages = [
            {"role": "system", "content": "You are a Personal Information Organizer..."},
            {"role": "user", "content": "Input: Alice prefers TypeScript\nOld Memory: []"},
        ]

        has_system = any(m.get("role") == "system" for m in fact_extraction_messages)
        assert has_system, (
            "INVARIANT BROKEN: Fact extraction messages must contain a system message. "
            "If mem0ai changed this, our schema detection needs updating."
        )

    def test_memory_update_no_system_message(self):
        """Memory update messages should NOT have role=system."""
        # Simulate what mem0ai sends for memory update decisions
        memory_update_messages = [
            {"role": "user", "content": "Existing Memories:\n...\nNew Memory: ..."},
        ]

        has_system = any(m.get("role") == "system" for m in memory_update_messages)
        assert not has_system, (
            "INVARIANT BROKEN: Memory update messages must NOT contain a system message. "
            "If mem0ai changed this, our schema detection needs updating."
        )


class TestSchemaDetectionWithRealPrompts:
    """Validate with realistic prompt structures from mem0ai source."""

    def test_fact_extraction_prompt_structure(self):
        """The FACT_RETRIEVAL_PROMPT template produces system+user messages."""
        # From mem0/memory/main.py — the prompt is passed as system message
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Personal Information Organizer, specialized in accurately "
                    "storing facts, user memories, and preferences. Your primary role is to "
                    "extract relevant pieces of information from conversations and organize "
                    "them into distinct, manageable facts."
                ),
            },
            {
                "role": "user",
                "content": "Input: Alice prefers TypeScript for new projects\nOld Memory: []",
            },
        ]

        system_count = sum(1 for m in messages if m.get("role") == "system")
        assert system_count == 1

    def test_memory_update_prompt_structure(self):
        """The UPDATE_MEMORY_PROMPT template produces user-only messages."""
        messages = [
            {
                "role": "user",
                "content": (
                    "Existing Memories:\n"
                    "---\n"
                    "ID: abc123\nMemory: Alice likes Python\n"
                    "---\n"
                    "New Memory: Alice now prefers TypeScript over Python\n"
                ),
            },
        ]

        system_count = sum(1 for m in messages if m.get("role") == "system")
        assert system_count == 0


class TestV2PromptShift:
    """Document why has_system alone is insufficient under mem0ai 2.x.

    Both the legacy 1.x fact-extraction prompt and the 2.x 'Memory Extractor'
    prompt carry a system message, so has_system cannot tell them apart — yet
    they require different output schemas. This is the regression that made
    add_memory return {"results": []} while every external dependency (Ollama,
    Qdrant, Anthropic) responded 200 OK.
    """

    def test_v2_memory_extractor_also_has_system_message(self):
        """The 2.x extraction prompt has a system message too — has_system ties."""
        messages = [
            {
                "role": "system",
                "content": (
                    "# ROLE\n\nYou are a Memory Extractor — a precise, "
                    "evidence-bound processor responsible for extracting rich, "
                    "contextual memories from conversations.\n\n"
                    '# OUTPUT FORMAT\n\nReturn ONLY valid JSON: {"memory": [...]}'
                ),
            },
            {"role": "user", "content": "## New Messages\nuser: Alice prefers TypeScript"},
        ]

        has_system = any(m.get("role") == "system" for m in messages)
        assert has_system, (
            "The 2.x 'Memory Extractor' prompt carries a system message just "
            "like the 1.x prompt — proving has_system cannot distinguish them."
        )

    def test_v2_prompt_requests_memory_not_facts(self):
        """The 2.x prompt's output contract is {"memory": [...]}, not {"facts": [...]}."""
        v2_system_prompt = (
            "You are a Memory Extractor ... "
            '# OUTPUT FORMAT\n\nReturn ONLY valid JSON: {"memory": [...]}'
        )
        assert '"memory"' in v2_system_prompt
        assert '"facts"' not in v2_system_prompt

"""
LangChain + Groq health article simplification and medical glossary engine.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


LEVEL_GUIDANCE = {
    "Everyday Reader (Grade 6–8)": (
        "Write for an everyday adult reader (6th to 8th grade reading level) "
        "using short, clear sentences and plain everyday words."
    ),
    "Very Simple (Grade 4–5)": (
        "Write in very simple, short sentences (4th to 5th grade reading level) "
        "that anyone can understand on a first read."
    ),
    "Patient & Caregiver Summary": (
        "Write a clear patient and caregiver summary organized around what the condition is, "
        "key numbers/thresholds, treatments, and safety warnings."
    ),
}

SYSTEM_PROMPT = """You are a careful medical communication specialist.
Your task is to simplify a complex health article into the requested target language while keeping EVERY clinical fact strictly unchanged.

Strict Rules:
1. NEVER change, round away, or invent dosages, lab thresholds, blood pressure numbers, percentages, timeframes, contraindications, or safety warnings.
2. NEVER add medical advice, diagnoses, or claims that are not present in the source article.
3. Write `simplified_title`, `simplified_article`, `key_facts_preserved`, and glossary `simple_meaning`/`context` in the requested Target Language.
4. In `glossary`, keep the original medical term in `term` (you may add its local-language equivalent in parentheses) so readers can recognize it in the source article.
5. Extract 5 to 12 of the most important complex medical terms from the article for the glossary.
6. Return ONLY a valid JSON object matching this exact schema (no markdown fences, no extra text):
{{
  "simplified_title": "string",
  "simplified_article": ["paragraph 1", "paragraph 2", "paragraph 3"],
  "key_facts_preserved": ["exact clinical fact/number/warning 1", "exact clinical fact/number/warning 2"],
  "glossary": [
    {{
      "term": "Original medical term",
      "simple_meaning": "Plain-language explanation in target language",
      "context": "Brief note on how this term is used in the article"
    }}
  ]
}}"""

USER_PROMPT = """Target Language: {language}
Style Guidance: {style_guidance}

Source Health Article:
{article_text}"""


def _parse_json_response(raw_text: str) -> Dict[str, Any]:
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end > start:
            data = json.loads(cleaned[start : end + 1])
        else:
            raise ValueError("Model returned invalid JSON. Please try clicking Simplify again.")

    if not isinstance(data, dict):
        raise ValueError("Unexpected response format from model.")

    paragraphs = data.get("simplified_article", [])
    if isinstance(paragraphs, str):
        paragraphs = [p.strip() for p in paragraphs.split("\n\n") if p.strip()]

    facts = data.get("key_facts_preserved", [])
    if isinstance(facts, str):
        facts = [facts]

    glossary = data.get("glossary", [])
    if not isinstance(glossary, list):
        glossary = []

    return {
        "simplified_title": str(data.get("simplified_title") or "Simplified Health Article"),
        "simplified_article": [str(p) for p in paragraphs if str(p).strip()],
        "key_facts_preserved": [str(f) for f in facts if str(f).strip()],
        "glossary": [
            {
                "term": str(item.get("term", "")).strip(),
                "simple_meaning": str(item.get("simple_meaning", "")).strip(),
                "context": str(item.get("context", "")).strip(),
            }
            for item in glossary
            if isinstance(item, dict) and item.get("term")
        ],
    }


def simplify_article(
    article_text: str,
    api_key: str,
    model_name: str,
    language: str,
    reading_level: str,
    max_chars: int = 16000,
) -> Tuple[Dict[str, Any], bool]:
    """Run the LangChain + Groq simplification pipeline and return (result_dict, was_truncated)."""
    was_truncated = len(article_text) > max_chars
    trimmed_text = article_text[:max_chars]

    llm = ChatGroq(
        api_key=api_key,
        model=model_name,
        temperature=0.2,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", USER_PROMPT),
        ]
    )

    chain = prompt | llm
    response = chain.invoke(
        {
            "language": language,
            "style_guidance": LEVEL_GUIDANCE.get(reading_level, LEVEL_GUIDANCE["Everyday Reader (Grade 6–8)"]),
            "article_text": trimmed_text,
        }
    )

    parsed = _parse_json_response(str(response.content))
    return parsed, was_truncated

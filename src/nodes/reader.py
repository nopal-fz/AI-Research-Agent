import json
import logging
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from src.state import ResearchState

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0
)

reader_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an AI research assistant.

Your task is to summarize a search result based on its title, URL, and snippet.

Return ONLY valid JSON with this structure:
{{
  "summary": "...",
  "key_points": ["...", "..."],
  "relevance_score": 0.0
}}

Rules:
- summary must be concise
- key_points must contain 2-4 points
- relevance_score must be a number between 0 and 1
- do not add markdown
- do not add extra explanation
"""
        ),
        (
            "human",
            """
Research topic:
{topic}

Search result:
Title: {title}
URL: {url}
Snippet: {snippet}
"""
        )
    ]
)

# This fallback is used when the LLM output cannot be parsed as JSON.
def _fallback_summary(result: dict[str, Any]) -> dict[str, Any]:
    """
    Create fallback summary when LLM output cannot be parsed.
    """
    return {
        "title": result.get("title", ""),
        "url": result.get("url", ""),
        "summary": result.get("snippet", ""),
        "key_points": [
            result.get("snippet", "")
        ],
        "relevance_score": 0.5
    }

# Define the reader node function
def reader_node(state: ResearchState) -> dict:
    """
    Summarize search results into source summaries.
    """
    logger.info("Running reader node")

    search_results = state.search_results

    if not search_results:
        logger.warning("No search results found. Skipping reader node.")
        return {"source_summaries": []}

    chain = reader_prompt | llm

    source_summaries: list[dict[str, Any]] = []

    # Limit the number of sources to summarize for efficiency
    max_sources = 5

    for idx, result in enumerate(search_results[:max_sources], start=1):
        logger.info(
            "Summarizing source %s/%s: %s",
            idx,
            min(len(search_results), max_sources),
            result.get("title", "")
        )

        response = chain.invoke(
            {
                "topic": state.topic,
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "snippet": result.get("snippet", "")
            }
        )

        logger.debug("Raw reader response: %s", response.content)

        try:
            parsed = json.loads(response.content)

            summary = {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "summary": parsed.get("summary", ""),
                "key_points": parsed.get("key_points", []),
                "relevance_score": parsed.get("relevance_score", 0.0)
            }

        except json.JSONDecodeError:
            logger.warning(
                "Reader response for source %s is not valid JSON. Using fallback summary.",
                idx
            )
            summary = _fallback_summary(result)

        source_summaries.append(summary)

    logger.info("Reader generated %s source summaries", len(source_summaries))

    return {
        "source_summaries": source_summaries
    }
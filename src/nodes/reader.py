import json
import logging
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from sentence_transformers import CrossEncoder

from src.state import ResearchState

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0
)

reranker = CrossEncoder(RERANKER_MODEL)

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
  "key_points": ["...", "..."]
}}

Rules:
- summary must be concise
- key_points must contain 2-4 points
- do not add relevance_score because it is calculated by the reranker
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

# Function to normalize raw cross-encoder scores into a 0-1 range using sigmoid
def normalize_score(raw_score: float) -> float:
    """
    Normalize raw cross-encoder score into 0-1 range using sigmoid.
    """

    normalized_score = 1 / (1 + pow(2.71828, -raw_score))
    return round(float(normalized_score), 4)

# Reranking function to be used in the reader node
def rerank_search_results(
    topic: str,
    search_results: list[dict[str, Any]],
    top_k: int = 5
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Rerank search results using a cross-encoder model.

    The reranker compares the research topic with each search result text.
    It returns the top_k most relevant sources and runtime relevance metrics.
    """

    logger.info("Running cross-encoder reranker")

    if not search_results:
        logger.warning("No search results available for reranking")

        return [], {
            "total_search_results": 0,
            "selected_sources": 0,
            "avg_relevance_score": 0.0,
            "max_relevance_score": 0.0,
            "min_relevance_score": 0.0,
            "reranker_model": RERANKER_MODEL,
        }

    pairs = []

    for result in search_results:
        document_text = f"{result.get('title', '')}\n{result.get('snippet', '')}"
        pairs.append((topic, document_text))

    raw_scores = reranker.predict(pairs)

    reranked_results = []

    for result, raw_score in zip(search_results, raw_scores):
        raw_score = float(raw_score)
        relevance_score = normalize_score(raw_score)

        reranked_results.append(
            {
                **result,
                "relevance_score": relevance_score,
                "raw_relevance_score": round(raw_score, 4),
            }
        )

    reranked_results = sorted(
        reranked_results,
        key=lambda x: x.get("relevance_score", 0),
        reverse=True
    )

    selected_results = reranked_results[:top_k]

    selected_scores = [
        item.get("relevance_score", 0.0)
        for item in selected_results
    ]

    metrics = {
        "total_search_results": len(search_results),
        "selected_sources": len(selected_results),
        "avg_relevance_score": round(
            sum(selected_scores) / len(selected_scores),
            4
        ) if selected_scores else 0.0,
        "max_relevance_score": max(selected_scores) if selected_scores else 0.0,
        "min_relevance_score": min(selected_scores) if selected_scores else 0.0,
        "reranker_model": RERANKER_MODEL,
    }

    logger.info("Reranker metrics: %s", metrics)

    return selected_results, metrics

# Fallback summary function in case reader LLM output cannot be parsed
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
        "relevance_score": result.get("relevance_score", 0.0),
        "raw_relevance_score": result.get("raw_relevance_score", 0.0),
    }

# Reader node function to be used in the research graph workflow
def reader_node(state: ResearchState) -> dict:
    """
    Rerank search results and summarize the most relevant sources.
    """

    logger.info("Running reader node")

    search_results = state.search_results

    if not search_results:
        logger.warning("No search results found. Skipping reader node.")

        return {
            "source_summaries": [],
            "research_metrics": {
                **state.research_metrics,
                "total_search_results": 0,
                "selected_sources": 0,
                "avg_relevance_score": 0.0,
                "max_relevance_score": 0.0,
                "min_relevance_score": 0.0,
                "reranker_model": RERANKER_MODEL,
            }
        }

    top_results, reranker_metrics = rerank_search_results(
        topic=state.topic,
        search_results=search_results,
        top_k=5
    )

    chain = reader_prompt | llm

    source_summaries: list[dict[str, Any]] = []

    for idx, result in enumerate(top_results, start=1):
        logger.info(
            "Summarizing source %s/%s | score=%s | title=%s",
            idx,
            len(top_results),
            result.get("relevance_score", 0.0),
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
                "relevance_score": result.get("relevance_score", 0.0),
                "raw_relevance_score": result.get("raw_relevance_score", 0.0),
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
        "source_summaries": source_summaries,
        "research_metrics": {
            **state.research_metrics,
            **reranker_metrics
        }
    }
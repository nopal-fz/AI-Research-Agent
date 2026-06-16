import logging
import os
from typing import Any

from dotenv import load_dotenv

from src.state import ResearchState

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# convert environment variables to correct types with safe defaults
def safe_int(value: Any, default: int = 0) -> int:
    """
    Safely convert a value to integer.
    """

    try:
        return int(value)
    except (TypeError, ValueError):
        return default

# convert environment variables to correct types with safe defaults
def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float.
    """

    try:
        return float(value)
    except (TypeError, ValueError):
        return default

MIN_SELECTED_SOURCES = safe_int(
    os.getenv("MIN_SELECTED_SOURCES"),
    default=3
)

MIN_AVG_RELEVANCE = safe_float(
    os.getenv("MIN_AVG_RELEVANCE"),
    default=0.70
)

MAX_ITERATIONS = safe_int(
    os.getenv("MAX_ITERATIONS"),
    default=2
)

# function is defined here to avoid circular import with graph.py
def critic_node(state: ResearchState) -> dict:
    """
    Evaluate whether collected sources are sufficient.
    If not sufficient, generate additional search queries and loop back to searcher.
    """

    logger.info("Running critic node")

    metrics = state.research_metrics or {}

    selected_sources = safe_int(metrics.get("selected_sources", 0))
    avg_relevance_score = safe_float(metrics.get("avg_relevance_score", 0.0))
    min_relevance_score = safe_float(metrics.get("min_relevance_score", 0.0))
    iteration = safe_int(state.iteration, 0) + 1

    logger.info(
        "Critic check | iteration=%s | selected_sources=%s | avg_score=%s | min_score=%s",
        iteration,
        selected_sources,
        avg_relevance_score,
        min_relevance_score
    )

    missing_points = []

    if selected_sources < MIN_SELECTED_SOURCES:
        missing_points.append("Not enough relevant sources were selected.")

    if avg_relevance_score < MIN_AVG_RELEVANCE:
        missing_points.append("Average relevance score is below the minimum threshold.")

    if not state.source_summaries:
        missing_points.append("No source summaries were generated.")

    if iteration >= MAX_ITERATIONS:
        logger.info("Maximum critic iterations reached. Proceeding to writer.")

        return {
            "is_sufficient": True,
            "missing_points": missing_points,
            "iteration": iteration
        }

    if missing_points:
        logger.warning("Evidence is not sufficient: %s", missing_points)

        additional_queries = [
            f"{state.topic} reliable sources",
            f"{state.topic} recent research findings",
            f"{state.topic} expert analysis"
        ]

        updated_queries = list(
            dict.fromkeys(state.search_queries + additional_queries)
        )

        return {
            "is_sufficient": False,
            "missing_points": missing_points,
            "search_queries": updated_queries,
            "iteration": iteration
        }

    logger.info("Evidence is sufficient")

    return {
        "is_sufficient": True,
        "missing_points": [],
        "iteration": iteration
    }
import logging
from typing import Any

from ddgs import DDGS

from src.state import ResearchState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the searcher node function
def searcher_node(state: ResearchState) -> dict:
    """
    Search web results based on generated search queries.
    """
    logger.info("Running searcher node")

    search_queries = state.search_queries

    if not search_queries:
        logger.warning("No search queries found. Skipping searcher node.")
        return {"search_results": []}

    all_results: list[dict[str, Any]] = []
    max_results_per_query = 5

    try:
        with DDGS() as ddgs:
            for query in search_queries:
                logger.info("Searching query: %s", query)

                results = ddgs.text(
                    query,
                    max_results=max_results_per_query
                )

                for result in results:
                    all_results.append(
                        {
                            "query": query,
                            "title": result.get("title", ""),
                            "url": result.get("href", ""),
                            "snippet": result.get("body", "")
                        }
                    )

        logger.info("Searcher collected %s total results", len(all_results))

    except Exception as e:
        logger.exception("Search failed: %s", e)
        return {"search_results": []}

    return {
        "search_results": all_results
    }
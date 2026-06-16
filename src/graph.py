import logging
from typing import Any

from langgraph.graph import StateGraph, START, END

from src.state import ResearchState
from src.nodes.validator import validator_node
from src.nodes.planner import planner_node
from src.nodes.searcher import searcher_node
from src.nodes.reader import reader_node
from src.nodes.critic import critic_node
from src.nodes.writer import writer_node

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Note: dotenv is loaded in app.py, so we can assume environment variables are available when this graph is built.
def get_state_value(state: Any, key: str, default: Any = None) -> Any:
    """
    Get value from either Pydantic state or dictionary state.
    """

    if hasattr(state, key):
        return getattr(state, key)

    if isinstance(state, dict):
        return state.get(key, default)

    return default

# The graph building function is defined at the end of this file to avoid circular imports with nodes.
def route_after_validation(state: ResearchState) -> str:
    """
    Decide whether to continue research or stop early after validation.
    """

    is_valid_topic = get_state_value(state, "is_valid_topic", False)

    logger.info("Routing after validation | is_valid_topic=%s", is_valid_topic)

    if is_valid_topic:
        return "planner"

    return "writer"

# route for critic node is defined here to avoid circular import with critic.py
def route_after_critic(state: ResearchState) -> str:
    """
    Decide whether to continue searching or write final report.
    """

    is_sufficient = get_state_value(state, "is_sufficient", False)

    logger.info("Routing after critic | is_sufficient=%s", is_sufficient)

    if is_sufficient:
        return "writer"

    return "searcher"

# The graph building function is defined at the end of this file to avoid circular imports with nodes.
def build_research_graph():
    """
    Build and compile the AI research agent workflow.

    Flow:
    START -> validator
    validator -> planner or writer
    planner -> searcher -> reader -> critic
    critic -> searcher or writer
    writer -> END
    """

    logger.info("Building research graph")

    graph = StateGraph(ResearchState)

    graph.add_node("validator", validator_node)
    graph.add_node("planner", planner_node)
    graph.add_node("searcher", searcher_node)
    graph.add_node("reader", reader_node)
    graph.add_node("critic", critic_node)
    graph.add_node("writer", writer_node)

    graph.add_edge(START, "validator")

    graph.add_conditional_edges(
        "validator",
        route_after_validation,
        {
            "planner": "planner",
            "writer": "writer"
        }
    )

    graph.add_edge("planner", "searcher")
    graph.add_edge("searcher", "reader")
    graph.add_edge("reader", "critic")

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "searcher": "searcher",
            "writer": "writer"
        }
    )

    graph.add_edge("writer", END)

    compiled_graph = graph.compile()

    logger.info("Research graph compiled successfully")

    return compiled_graph
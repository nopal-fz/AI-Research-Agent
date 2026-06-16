import logging

from langgraph.graph import StateGraph, START, END

from src.state import ResearchState
from src.nodes.planner import planner_node
from src.nodes.searcher import searcher_node
from src.nodes.reader import reader_node
from src.nodes.writer import writer_node

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# This function builds the research graph by defining the nodes and edges of the workflow.
def build_research_graph():
    """
    Build and compile the AI research agent workflow.

    Flow:
    START -> planner -> searcher -> reader -> writer -> END
    """
    logger.info("Building research graph")

    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner_node)
    graph.add_node("searcher", searcher_node)
    graph.add_node("reader", reader_node)
    graph.add_node("writer", writer_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "searcher")
    graph.add_edge("searcher", "reader")
    graph.add_edge("reader", "writer")
    graph.add_edge("writer", END)

    compiled_graph = graph.compile()

    logger.info("Research graph compiled successfully")

    return compiled_graph
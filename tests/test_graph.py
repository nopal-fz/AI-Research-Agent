from src.graph import build_research_graph

# Test to validate that the research graph can be built and compiled successfully
def test_graph_can_compile():
    graph = build_research_graph()

    assert graph is not None
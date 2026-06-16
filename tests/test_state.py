from src.state import ResearchState

# Test to validate that ResearchState initializes with correct default values
def test_research_state_default_values():
    state = ResearchState(topic="AI agents in data analytics")

    assert state.topic == "AI agents in data analytics"
    assert state.sub_questions == []
    assert state.search_queries == []
    assert state.search_results == []
    assert state.source_summaries == []
    assert state.research_metrics == {}
    assert state.final_report == ""
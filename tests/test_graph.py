from src.state import ResearchState

def test_research_state_can_be_initialized():
    state = ResearchState(topic="AI agents in data analytics")

    assert state.topic == "AI agents in data analytics"
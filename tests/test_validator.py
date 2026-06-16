from src.state import ResearchState
from src.nodes.validator import validator_node

def test_validator_rejects_noise_input():
    state = ResearchState(topic="hahahah wkwk")

    result = validator_node(state)

    assert result["is_valid_topic"] is False

def test_validator_accepts_valid_topic():
    state = ResearchState(topic="The impact of AI agents on data analyst jobs")

    result = validator_node(state)

    assert result["is_valid_topic"] is True
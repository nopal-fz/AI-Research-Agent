import logging
import re

from src.state import ResearchState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Performs valid input checks to prevent unnecessary graph execution on invalid topics.
def validator_node(state: ResearchState) -> dict:
    """
    Validate user research topic before entering the research workflow.
    Invalid topics will stop the graph before planner/searcher.
    """

    logger.info("Running validator node")

    topic = state.topic.strip()
    topic_lower = topic.lower()

    if not topic:
        logger.warning("Invalid topic: empty input")

        return {
            "is_valid_topic": False,
            "validation_message": "Please enter a research topic first.",
            "final_report": "Please enter a valid research topic before running the agent."
        }

    noise_patterns = [
        "asdf",
        "qwerty",
        "lorem ipsum",
        "test test",
        "wkwk",
        "wk wk",
        "haha",
        "hahaha",
        "hehe",
        "hihi",
        "xixixi",
        "lol",
        "random text",
    ]

    if any(pattern in topic_lower for pattern in noise_patterns):
        logger.warning("Invalid topic: noise pattern detected")

        return {
            "is_valid_topic": False,
            "validation_message": "The input appears to be noise or test text.",
            "final_report": "The input appears to be noise or test text. Please enter a real research topic."
        }

    if re.fullmatch(r"[\W_]+", topic):
        logger.warning("Invalid topic: symbol-only input")

        return {
            "is_valid_topic": False,
            "validation_message": "The input does not look like a valid research topic.",
            "final_report": "The input does not look like a valid research topic. Please enter a meaningful topic."
        }

    if re.fullmatch(r"(.)\1{5,}", topic_lower):
        logger.warning("Invalid topic: repeated character input")

        return {
            "is_valid_topic": False,
            "validation_message": "The input does not look like a valid research topic.",
            "final_report": "The input does not look like a valid research topic. Please enter a meaningful topic."
        }

    alphabetic_chars = re.findall(r"[a-zA-Z]", topic)
    if len(alphabetic_chars) < 8:
        logger.warning("Invalid topic: not enough alphabetic characters")

        return {
            "is_valid_topic": False,
            "validation_message": "The topic is too short or unclear.",
            "final_report": "The topic is too short or unclear. Please provide a more specific research topic."
        }

    if len(topic.split()) < 3:
        logger.warning("Invalid topic: too vague")

        return {
            "is_valid_topic": False,
            "validation_message": "The topic is too vague.",
            "final_report": "The topic is too vague. Please provide a clearer research topic."
        }

    logger.info("Topic passed validation")

    return {
        "is_valid_topic": True,
        "validation_message": "Topic is valid."
    }
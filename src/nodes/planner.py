import json
import os
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from src.state import ResearchState

import dotenv
dotenv.load_dotenv()  # Load environment variables from .env file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize ollama model from .env
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0
)

# Define prompt for the planner node
planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an AI research planner.

Your task is to break down the user's research topic into:
1. 3-5 clear sub-questions
2. 3-5 search queries for web research

Return ONLY valid JSON with this structure:
{{
  "sub_questions": ["..."],
  "search_queries": ["..."]
}}
Do not add markdown, explanation, or extra text.
"""
        ),
        (
            "human",
            "Research topic: {topic}"
        )
    ]
)

# Define the planner node function
def planner_node(state: ResearchState) -> dict:
    """
    Generate research sub-questions and search queries from the main topic.
    """
    logger.info("Running planner node")
    logger.info(f"Research topic: {state.topic}")
    
    chain = planner_prompt | llm

    response = chain.invoke({
        "topic": state.topic
    })

    logger.debug(f"Raw response from LLM: {response.content}")

    try:
        result = json.loads(response.content)
        
        sub_queries = result.get("sub_questions", [])
        search_queries = result.get("search_queries", [])
        
        logger.info(
            "Planner generated %s sub-questions and %s search queries",
            len(sub_queries),
            len(search_queries)
        )
        
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON from LLM response. Returning empty lists.")
        
        result = {
            "sub_questions": [
                f"What are the key concepts related to {state.topic}?",
                f"What are the benefits and challenges of {state.topic}?",
                f"What are recent trends related to {state.topic}?"
            ],
            "search_queries": [
                state.topic,
                f"{state.topic} benefits challenges",
                f"{state.topic} recent trends"
            ]
        }

    return {
        "sub_questions": result.get("sub_questions", []),
        "search_queries": result.get("search_queries", [])
    }
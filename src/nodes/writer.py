import logging
import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from src.state import ResearchState

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0.2
)

writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an AI research report writer.

Your task is to write a structured research report based on:
1. research topic
2. sub-questions
3. source summaries

Write the report in clear markdown format.

Report structure:
# Research Report: <topic>

## Executive Summary
Briefly explain the overall findings.

## Key Findings
Write 3-5 key findings.

## Detailed Analysis
Explain the topic based on the source summaries.

## Implications
Explain the practical implications.

## Limitations
Mention limitations of the current research based only on available snippets/summaries.

## References
List source title and URL.

Rules:
- Do not invent sources.
- Use only the provided source summaries.
- If evidence is limited, say so clearly.
- Keep the report concise but useful.
"""
        ),
        (
            "human",
            """
Research topic:
{topic}

Sub-questions:
{sub_questions}

Source summaries:
{source_summaries}
"""
        )
    ]
)

# Define the writer node function
def writer_node(state: ResearchState) -> dict:
    """
    Generate the final research report from source summaries.
    """

    logger.info("Running writer node")

    if not state.source_summaries:
        logger.warning("No source summaries found. Generating limited report.")

    chain = writer_prompt | llm

    response = chain.invoke(
        {
            "topic": state.topic,
            "sub_questions": state.sub_questions,
            "source_summaries": state.source_summaries
        }
    )

    final_report = response.content

    logger.info("Writer generated final report with %s characters", len(final_report))

    return {
        "final_report": final_report
    }
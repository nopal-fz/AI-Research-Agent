from typing import List, Dict, Any
from pydantic import BaseModel, Field

class ResearchState(BaseModel):
    topic: str = Field(
        "",
        description="The research topic or question"
    )
    is_valid_topic: bool = Field(
        True,
        description="Whether the user topic is valid for research"
    )
    validation_message: str = Field(
        "",
        description="Message explaining why the topic is invalid"
    )
    sub_questions: List[str] = Field(
        default_factory=list,
        description="List of sub-questions derived from the main topic"
    )
    search_queries: List[str] = Field(
        default_factory=list,
        description="List of search queries generated for information retrieval"
    )
    search_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of search results with metadata"
    )
    source_summaries: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Summaries extracted from each search result/source"
    )
    research_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Runtime metrics for the research workflow"
    )
    is_sufficient: bool = Field(
        False,
        description="Whether the collected evidence is sufficient"
    )
    missing_points: List[str] = Field(
        default_factory=list,
        description="Missing points identified by critic node"
    )
    iteration: int = Field(
        0,
        description="Number of research loop iterations"
    )
    final_report: str = Field(
        "",
        description="The final synthesized report based on the research findings"
    )
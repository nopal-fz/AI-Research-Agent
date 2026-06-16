from typing import List, Dict, Any
from pydantic import BaseModel, Field

class ResearchState(BaseModel):
    topic: str = Field(
        "",
        description="The research topic or question"
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
    final_report: str = Field(
        "",
        description="The final synthesized report based on the research findings"
    )
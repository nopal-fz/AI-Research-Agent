import logging

from src.graph import build_research_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

graph = build_research_graph() # Build the research graph workflow

result = graph.invoke(
    {
        "topic": "The impact of AI agents on data analyst jobs"
    }
)

print("\n===== FINAL REPORT =====\n")
print(result["final_report"])

print("\n===== DEBUG INFO =====\n")
print("Sub questions:", len(result["sub_questions"]))
print("Search queries:", len(result["search_queries"]))
print("Search results:", len(result["search_results"]))
print("Source summaries:", len(result["source_summaries"]))
print("Research metrics:", result["research_metrics"])
print("Is valid topic:", result["is_valid_topic"])
print("Validation message:", result["validation_message"])
print("Is sufficient:", result["is_sufficient"])
print("Missing points:", result["missing_points"])
print("Iteration:", result["iteration"])
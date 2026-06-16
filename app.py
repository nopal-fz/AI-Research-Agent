import logging

import streamlit as st
from dotenv import load_dotenv

from src.graph import build_research_graph
from src.report import generate_pdf


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 AI Research Agent")
st.write(
    "An AI research assistant built with LangGraph, LangSmith, Ollama, and a cross-encoder reranker."
)


def get_graph():
    """
    Build research graph.

    Cache is intentionally disabled during development to avoid using stale graph logic.
    After the workflow is stable, this can be changed to st.cache_resource.
    """

    return build_research_graph()


topic = st.text_input(
    "Enter research topic",
    placeholder="Example: The impact of AI agents on data analyst jobs"
)

run_button = st.button("Run Research Agent")

if run_button:
    if not topic.strip():
        st.warning("Please enter a research topic first.")
    else:
        with st.spinner("Running research workflow..."):
            graph = get_graph()

            result = graph.invoke(
                {
                    "topic": topic
                },
                config={
                    "tags": ["streamlit", "mvp", "reranker", "conditional-edge"],
                    "metadata": {
                        "app": "ai-research-agent",
                        "search_provider": "ddgs",
                        "workflow": "validator-planner-searcher-reader-critic-writer"
                    }
                }
            )

        if not result.get("is_valid_topic", True):
            st.warning(result.get("validation_message", "Invalid research topic."))
            st.info(result.get("final_report", "Please enter a valid research topic."))

        else:
            st.success("Research completed!")

            metrics = result.get("research_metrics", {})

            st.subheader("Research Metrics")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Search Results",
                metrics.get("total_search_results", 0)
            )

            col2.metric(
                "Selected Sources",
                metrics.get("selected_sources", 0)
            )

            col3.metric(
                "Avg Relevance",
                metrics.get("avg_relevance_score", 0)
            )

            col4.metric(
                "Max Relevance",
                metrics.get("max_relevance_score", 0)
            )

            st.subheader("Final Report")
            st.markdown(result["final_report"])

            # Provide PDF download of the final report (ReportLab)
            try:
                # Generate PDF once and store in session_state to avoid
                # re-running the whole workflow when the download button is clicked.
                pdf_bytes = generate_pdf(topic, result)
                safe_name = (topic or "report").strip().replace(" ", "_")[:60]
                st.session_state["last_pdf"] = pdf_bytes
                st.session_state["last_pdf_name"] = f"{safe_name}.pdf"
            except Exception as e:
                st.error(f"Failed to generate PDF: {e}")

# Persistent download button: shown when a PDF was previously generated
if st.session_state.get("last_pdf"):
    try:
        st.download_button(
            label="Download last PDF",
            data=st.session_state.get("last_pdf"),
            file_name=st.session_state.get("last_pdf_name"),
            mime="application/pdf",
            key="download_report_persistent",
        )
    except Exception as e:
        st.error(f"Failed to prepare download: {e}")

# Debug info: show immediately after a run when `result` exists in this run
if 'result' in globals():
    with st.expander("Debug Info"):
        st.write("Is valid topic:", result.get("is_valid_topic"))
        st.write("Validation message:", result.get("validation_message"))
        st.write("Sub questions:", result.get("sub_questions"))
        st.write("Search queries:", result.get("search_queries"))
        st.write("Search results:", result.get("search_results"))
        st.write("Source summaries:", result.get("source_summaries"))
        st.write("Research metrics:", result.get("research_metrics"))
        st.write("Is sufficient:", result.get("is_sufficient"))
        st.write("Missing points:", result.get("missing_points"))
        st.write("Iteration:", result.get("iteration"))
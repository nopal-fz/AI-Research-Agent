import logging

import streamlit as st
from dotenv import load_dotenv

from src.graph import build_research_graph

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

st.set_page_config(
    page_title="AI Research Agent",
    layout="wide"
)

# Streamlit app to run the AI research agent workflow
st.title("🔎 AI Research Agent")
st.write(
    "An AI research assistant built with LangGraph, LangSmith, and Ollama."
)

# Input for research topic
topic = st.text_input(
    "Enter research topic",
    placeholder="Example: The impact of AI agents on data analyst jobs"
)

# Button to run the research workflow
run_button = st.button("Run Research Agent")
if run_button:
    if not topic.strip():
        st.warning("Please enter a research topic first.")
    else:
        with st.spinner("Running research workflow..."):
            graph = build_research_graph()

            result = graph.invoke(
                {
                    "topic": topic
                }
            )

        st.success("Research completed!")

        st.subheader("Final Report")
        st.markdown(result["final_report"])

        with st.expander("Debug Info"):
            st.write("Sub questions:", result["sub_questions"])
            st.write("Search queries:", result["search_queries"])
            st.write("Search results:", result["search_results"])
            st.write("Source summaries:", result["source_summaries"])
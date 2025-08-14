# main.py

import os
import re
import time
from html import escape
from io import StringIO, BytesIO

import streamlit as st
from docx import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.vectorstores import FAISS

# from mysecrets import OPENAI_API_KEY
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")

from utils import (
    chunk_embed_store_transcript,
    build_retriever,
    get_llm_client,
    generate_insights,
    split_insights_into_points,
    find_supporting_quotes,
    export_to_word,
    extract_insight_summaries,
)

def main():
    st.markdown("<h1 style='font-weight:bold;'>💡Decode</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:gray; font-size:14px; margin-top:-10px;'>Qualitative insights you can trace, and trust</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "You may upload more than one", type=["docx", "txt"], accept_multiple_files=True
    )

    if uploaded_files:
        all_transcripts = [parse_transcript(f) for f in uploaded_files]

        # --- Research question ---
        st.subheader("Enter Your Research Question")
        placeholder_text = "What are you trying to find out from this transcript?"
        with st.form(key="rq_form", clear_on_submit=False):
            research_question = st.text_area("", placeholder=placeholder_text, height=80, key="rq_text")
            submitted = st.form_submit_button("Submit Research Question", type="primary")

        if submitted:
            if not research_question.strip():
                st.error("Please enter a research question before submitting.")
                st.stop()
            st.session_state["research_question"] = research_question.strip()

        if "research_question" in st.session_state:
            rq = st.session_state["research_question"]

            st.markdown(f"<em>{escape(rq)}</em>", unsafe_allow_html=True)

            # --- Build vector store in-memory ---
            if "vectordb" not in st.session_state:
                with st.spinner("Processing transcripts and generating embeddings..."):
                    vectordb, all_chunks, embeddings_model = chunk_embed_store_transcript(
                        "\n".join(all_transcripts)
                    ), st.session_state.get("all_chunks"), st.session_state.get("embeddings_model")

                    # Store in session
                    st.session_state["vectordb"] = vectordb
                    st.session_state["all_chunks"] = all_chunks
                    st.session_state["embeddings_model"] = embeddings_model
                    st.session_state["all_transcripts"] = all_transcripts

            all_chunks = st.session_state["all_chunks"]
            embeddings_model = st.session_state["embeddings_model"]

            # Build retriever using in-memory FAISS instance
            retriever = vectordb.as_retriever(search_kwargs={"k": 8})
            client = get_llm_client()
            relevant_docs = retriever.get_relevant_documents(rq)

            with st.spinner("Generating insights, please wait..."):
                insight_text = generate_insights(client, rq, relevant_docs)
                insight_points = split_insights_into_points(insight_text)
                summaries = extract_insight_summaries(insight_points)
                supporting_quotes = find_supporting_quotes(insight_points, all_chunks, embeddings_model)

            # --- Findings display ---
            for i, point in enumerate(insight_points):
                summary = summaries[i].strip(" *")
                st.write(f"**Insight {i+1}: {summary}**")
                st.write(point)
                st.write("Supporting quotes:")
                for quote in supporting_quotes[i]:
                    st.markdown(f"- {quote}")

            # --- Export Word ---
            if st.button("Download Findings as Word Document"):
                doc_stream = export_to_word([(rq, insight_text)], [supporting_quotes])
                st.download_button(
                    label="📥 Click to Download",
                    data=doc_stream,
                    file_name="decode_findings.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )


if __name__ == "__main__":
    main()

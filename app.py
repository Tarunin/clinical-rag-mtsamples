import streamlit as st
import sys
import os

st.set_page_config(
    page_title="Clinical Note QA | MIMIC RAG",
    page_icon="🏥",
    layout="wide"
)

with st.sidebar:
    st.title("⚙️ Settings")
    top_k = st.slider("Sources to retrieve", min_value=1, max_value=10, value=5)
    st.markdown("---")
    st.markdown("""
    **About this project**  
    RAG pipeline over MTSamples medical transcription notes.  
    
    - **Embeddings:** `all-MiniLM-L6-v2`  
    - **Vector Store:** FAISS (IndexFlatIP)  
    - **LLM:** LLaMA 3.3 70B via Groq  
    - **Dataset:** MTSamples (4000+ clinical transcriptions)  
    
    ⚠️ *For research use only. Not for clinical decisions.*
    """)
    st.markdown("---")
    st.markdown("[GitHub](https://github.com/Tarunin/mimic-clinical-rag) | [MTSamples](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions)")

@st.cache_resource(show_spinner="Loading index and models...")
def load_pipeline():
    import rag_pipeline as rag  # imports 3_rag_pipeline.py renamed to rag_pipeline.py
    return rag

try:
    rag = load_pipeline()
    pipeline_ready = True
except Exception as e:
    st.error(f"Failed to load pipeline: {e}")
    pipeline_ready = False

st.title("🏥 Clinical Note QA")
st.caption("Retrieval-Augmented Generation over MTSamples Medical Transcriptions")
st.markdown("---")

st.markdown("**Try an example query:**")
examples = [
    "What medications were prescribed to the patient?",
    "What was the primary diagnosis?",
    "Describe the surgical procedure performed.",
    "What were the patient's presenting symptoms?",
    "What follow-up care was recommended?",
]
cols = st.columns(len(examples))
selected_example = None
for i, (col, ex) in enumerate(zip(cols, examples)):
    if col.button(f"Ex {i+1}", help=ex, use_container_width=True):
        selected_example = ex


query_input = st.text_area(
    "Enter your clinical question:",
    value=selected_example if selected_example else "",
    height=80,
    placeholder="e.g. What medications was the patient prescribed at discharge?"
)

search_btn = st.button("🔍 Search", type="primary", disabled=not pipeline_ready)

if search_btn and query_input.strip():
    with st.spinner("Retrieving and generating answer..."):
        try:
            answer_text, sources = rag.answer(query_input.strip(), top_k=top_k)

            
            st.markdown("### 💬 Answer")
            st.success(answer_text)

            # Sources
            st.markdown(f"### 📄 Retrieved Sources ({len(sources)})")
            for i, src in enumerate(sources, 1):
                score_pct = f"{src['score']*100:.1f}%"
                with st.expander(
                    f"Source {i} — {src['subject_id']} | "
                    f"Specialty: {src['hadm_id']} | Similarity: {score_pct}"
                ):
                    st.text(src["text"])

        except Exception as e:
            st.error(f"Error during inference: {e}")

elif search_btn and not query_input.strip():
    st.warning("Please enter a question first.")

st.markdown("---")
st.caption(
    "Built with MTSamples medical transcription dataset, "
    "FAISS, sentence-transformers, LLaMA 3.3 70B via Groq. "
    "For research purposes only."
)

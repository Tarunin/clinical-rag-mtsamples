# 🏥 Clinical Note QA — RAG over MIMIC-III/IV

A **Retrieval-Augmented Generation (RAG)** system for querying clinical discharge summaries using natural language. Built on MIMIC-III/IV EHR data with a FAISS vector store and LLaMA 3 via Groq.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS-orange)
![LLM](https://img.shields.io/badge/LLM-LLaMA%203%208B-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

---

## 📌 Overview

Clinical notes in EHR systems contain rich, unstructured information that is difficult to query at scale. This project builds a full RAG pipeline that:

1. **Chunks** 50K+ MIMIC discharge summaries into overlapping segments
2. **Embeds** them using `sentence-transformers/all-MiniLM-L6-v2` and stores in a FAISS index
3. **Retrieves** the most semantically relevant chunks for a natural language query
4. **Generates** grounded, cited answers using LLaMA 3 8B (via Groq API)
5. **Serves** everything through an interactive Streamlit UI

---

## 🏗️ Architecture

```
Query
  │
  ▼
Sentence Embedding (all-MiniLM-L6-v2)
  │
  ▼
FAISS Similarity Search (IndexFlatIP, cosine)
  │
  ▼
Top-K Retrieved Chunks (with patient/admission metadata)
  │
  ▼
LLaMA 3 8B via Groq (grounded generation with citations)
  │
  ▼
Answer + Source Snippets
```

---

## 📁 Project Structure

```
mimic-clinical-rag/
│
├── 1_preprocess.py        # Load MIMIC, clean PHI, chunk notes
├── 2_build_index.py       # Embed chunks, build FAISS index
├── rag_pipeline.py        # Core RAG: retrieve + generate
├── app.py                 # Streamlit UI
├── requirements.txt
├── data/                  # (gitignored — MIMIC access required)
│   ├── NOTEEVENTS.csv
│   ├── processed_notes.jsonl
│   ├── faiss_index.bin
│   └── metadata.pkl
└── README.md
```

---

## 🚀 Setup & Run

### 1. Prerequisites

- Python 3.10+
- [PhysioNet credentialed access](https://physionet.org/settings/credentialing/) to MIMIC-III or MIMIC-IV
- Free [Groq API key](https://console.groq.com) for LLM inference

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your data

Place your MIMIC data file at `data/NOTEEVENTS.csv` (MIMIC-III) or `data/discharge.csv` (MIMIC-IV).

### 4. Run the pipeline

```bash
# Step 1: Preprocess and chunk notes
python 1_preprocess.py

# Step 2: Embed and build FAISS index
python 2_build_index.py

# Step 3: Launch the app
export GROQ_API_KEY=your_key_here
streamlit run app.py
```

---

## 💬 Example Queries

| Query | What it tests |
|---|---|
| *What medications was the patient prescribed at discharge?* | Medication extraction |
| *What was the primary diagnosis for this admission?* | Diagnosis retrieval |
| *Did the patient have a history of diabetes?* | Medical history lookup |
| *What follow-up instructions were given?* | Discharge planning |
| *Were there any complications during the stay?* | Adverse event detection |

---

## 🔧 Tech Stack

| Component | Tool |
|---|---|
| Dataset | MIMIC-III / MIMIC-IV (PhysioNet) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS (`IndexFlatIP`, cosine similarity) |
| LLM | LLaMA 3 8B via Groq API |
| UI | Streamlit |
| Language | Python 3.10+ |

---

## ⚠️ Data Access & Ethics

This project uses [MIMIC-III](https://physionet.org/content/mimiciii/) and [MIMIC-IV](https://physionet.org/content/mimiciv/), which require credentialed PhysioNet access. The dataset contains de-identified patient data and must be used in accordance with the PhysioNet data use agreement.

**This system is for research purposes only and is not intended for clinical decision-making.**

---

## 👤 Author

**Tarun Sethi**  
MS Data Science, Northeastern University  
[LinkedIn](https://linkedin.com/in/sethi-tarun) | [GitHub](https://github.com/Tarunin)

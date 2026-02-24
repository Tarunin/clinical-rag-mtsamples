import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq        
import os

INDEX_PATH   = "data/faiss_index.bin"
META_PATH    = "data/metadata.pkl"
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL    = "llama-3.3-70b-versatile" 
TOP_K        = 5         

SYSTEM_PROMPT = """You are a clinical informatics assistant. You answer questions 
about patient discharge summaries strictly based on the provided context. 
If the context does not contain enough information to answer, say so clearly.
Never fabricate clinical details. Always cite which patient/admission the 
information comes from."""

print("Loading FAISS index and metadata...")
index    = faiss.read_index(INDEX_PATH)

with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)

print(f"Index loaded: {index.ntotal} vectors")
print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)
print("Embedding model loaded")

client = Groq(api_key=os.environ["GROQ_API_KEY"])

def retrieve(query: str, top_k: int = TOP_K):
    query_vec = embedder.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    distances, indices = index.search(query_vec, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        chunk = metadata[idx].copy()
        chunk["score"] = float(dist)
        results.append(chunk)

    return results

def build_context(chunks):
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[Source {i} | Patient: {c['subject_id']} | Admission: {c['hadm_id']}]\n{c['text']}"
        )
    return "\n\n---\n\n".join(parts)

def answer(query: str, top_k: int = TOP_K):
    retrieved_chunks = retrieve(query, top_k)

    if not retrieved_chunks:
        return "No relevant documents found.", []

    context = build_context(retrieved_chunks)

    user_message = f"""Context from clinical discharge summaries:

{context}

---

Question: {query}

Answer based only on the context above. Cite which source(s) you used."""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": user_message}
        ],
        temperature=0.2,        # Low temp for factual clinical answers
        max_tokens=512,
    )

    answer_text = response.choices[0].message.content
    return answer_text, retrieved_chunks


if __name__ == "__main__":
    q = "What medications was the patient prescribed at discharge?"
    ans, sources = answer(q)
    print(f"\nQ: {q}")
    print(f"\nA: {ans}")
    print(f"\nRetrieved {len(sources)} sources")

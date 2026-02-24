"""
Step 2: Embed chunks and build FAISS index
Run: python 2_build_index.py
Uses sentence-transformers (free, local, no API key needed)
"""

import json
import faiss
import numpy as np
import pickle
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ── CONFIG ─────────────────────────────────────────────────────────────────────
CHUNKS_PATH  = "data/processed_notes.jsonl"
INDEX_PATH   = "data/faiss_index.bin"
META_PATH    = "data/metadata.pkl"
MODEL_NAME   = "sentence-transformers/all-MiniLM-L6-v2"  # Fast + good quality
BATCH_SIZE   = 64

# ── LOAD CHUNKS ────────────────────────────────────────────────────────────────
print("Loading chunks...")
chunks    = []
metadata  = []

with open(CHUNKS_PATH) as f:
    for line in f:
        rec = json.loads(line)
        chunks.append(rec["text"])
        metadata.append({
            "subject_id": rec["subject_id"],
            "hadm_id":    rec["hadm_id"],
            "chunk_id":   rec["chunk_id"],
            "text":       rec["text"]
        })

print(f"Loaded {len(chunks)} chunks")

# ── EMBED ──────────────────────────────────────────────────────────────────────
print(f"Loading embedding model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

print("Embedding chunks (this takes a few minutes)...")
embeddings = model.encode(
    chunks,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True   # cosine similarity via inner product
)

embeddings = np.array(embeddings).astype("float32")
print(f"Embedding shape: {embeddings.shape}")

# ── BUILD FAISS INDEX ──────────────────────────────────────────────────────────
dim   = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)     # Inner product = cosine sim (since normalized)
index.add(embeddings)
print(f"FAISS index built with {index.ntotal} vectors (dim={dim})")

# ── SAVE ───────────────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
faiss.write_index(index, INDEX_PATH)

with open(META_PATH, "wb") as f:
    pickle.dump(metadata, f)

print(f"\nSaved index to {INDEX_PATH}")
print(f"Saved metadata to {META_PATH}")
print("\nDone! Ready to run the QA pipeline.")

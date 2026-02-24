"""
Step 1: Load and preprocess MIMIC discharge summaries
Run: python 1_preprocess.py
"""

import pandas as pd
import re
import os
import json
from tqdm import tqdm

# ── CONFIG ─────────────────────────────────────────────────────────────────────
NOTES_PATH = "data/mtsamples.csv"          # MIMIC-III
# NOTES_PATH = "data/discharge.csv"         # MIMIC-IV (uncomment if using IV)
OUTPUT_PATH = "data/processed_notes.jsonl"
MAX_NOTES   = 5000   # Start with 5k for speed; scale up later
CHUNK_SIZE  = 500    # tokens (approx chars / 4)
CHUNK_OVERLAP = 50   # overlap between chunks for context continuity

# ── LOAD ───────────────────────────────────────────────────────────────────────
print("Loading notes...")
df = pd.read_csv(NOTES_PATH, nrows=MAX_NOTES, low_memory=False)

df.columns = [c.lower() for c in df.columns]

text_col = 'transcription'
id_col   = 'sample_name'
adm_col  = 'medical_specialty'

df = df[[id_col, adm_col, text_col]].dropna(subset=[text_col])
print(f"Loaded {len(df)} discharge notes")

# ── CLEAN ──────────────────────────────────────────────────────────────────────
def clean_note(text):
    text = re.sub(r'\[\*\*.*?\*\*\]', '[DEIDENTIFIED]', text)   # remove PHI placeholders
    text = re.sub(r'\n{3,}', '\n\n', text)                       # collapse excess newlines
    text = re.sub(r'[ \t]{2,}', ' ', text)                       # collapse spaces
    return text.strip()

# ── CHUNK ──────────────────────────────────────────────────────────────────────
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping word-based chunks."""
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

# ── PROCESS & SAVE ─────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
total_chunks = 0

with open(OUTPUT_PATH, "w") as f:
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing notes"):
        cleaned  = clean_note(str(row[text_col]))
        chunks   = chunk_text(cleaned)
        for i, chunk in enumerate(chunks):
            record = {
                "subject_id": str(row[id_col]),
                "hadm_id":    str(row.get(adm_col, "")),
                "chunk_id":   i,
                "text":       chunk
            }
            f.write(json.dumps(record) + "\n")
            total_chunks += 1

print(f"\nDone! Saved {total_chunks} chunks to {OUTPUT_PATH}")

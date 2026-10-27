"""Build (or rebuild) the index from corpus/docs.json.

Usage:  ~/workspace/ml_env/bin/python build_index.py
Output: data/chunks.json + data/index.faiss
"""

import json
from pathlib import Path

from rag_pipeline import DOCS_PATH, build_chunks, build_index

if __name__ == "__main__":
    docs = json.loads(Path(DOCS_PATH).read_text(encoding="utf-8"))
    print(f"Docs: {len(docs)}")
    chunks = build_chunks(docs)
    print(f"Chunks: {len(chunks)}")
    build_index(chunks)
    print("Index written to data/index.faiss, chunks to data/chunks.json")

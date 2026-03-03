import os
import hashlib
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX = os.environ["PINECONE_INDEX"]

DATA_DIR = Path("data/manual")

# 384-dim local model
MODEL_NAME = os.environ.get("LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def file_modified_iso(path: Path) -> str:
    return iso_z(datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc))

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks

def stable_doc_id(path: Path, modified_at: str) -> str:
    h = hashlib.sha256()
    h.update(str(path.resolve()).encode("utf-8"))
    h.update(modified_at.encode("utf-8"))
    return h.hexdigest()[:24]

def main():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    idx = pc.Index(PINECONE_INDEX)

    scope = os.environ.get("SCOPE", "personal").strip().lower()
    if scope not in {"personal", "work"}:
        raise SystemExit("SCOPE must be 'personal' or 'work'.")

    model = SentenceTransformer(MODEL_NAME)

    files = [p for p in DATA_DIR.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".md"}]
    if not files:
        print(f"No .txt/.md files found in {DATA_DIR.resolve()}")
        return

    total_chunks = 0
    for path in files:
        modified_at = file_modified_iso(path)
        doc_id = stable_doc_id(path, modified_at)
        title = path.name

        text = read_text(path)
        chunks = chunk_text(text)
        if not chunks:
            continue

        embeddings = model.encode(chunks, normalize_embeddings=True)

        vectors = []
        for i, emb in enumerate(embeddings):
            vector_id = f"{doc_id}:{i}"
            vectors.append({
                "id": vector_id,
                "values": emb.tolist(),
                "metadata": {
                    "scope": scope,
                    "source_system": "manual",
                    "title": title,
                    "deep_link": str(path.resolve()),
                    "modified_at": modified_at,
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "text": chunks[i][:2000],
                }
            })

        idx.upsert(vectors=vectors)
        total_chunks += len(vectors)
        print(f"Upserted {len(vectors):>3} chunks from {title} (scope={scope})")

    print(f"\nDone. Total chunks upserted: {total_chunks}")

if __name__ == "__main__":
    main()

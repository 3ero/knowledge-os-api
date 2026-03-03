import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

API_BEARER_TOKEN = os.environ.get("API_BEARER_TOKEN", "change_me")
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX = os.environ["PINECONE_INDEX"]
MODEL_NAME = os.environ.get("LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

pc = Pinecone(api_key=PINECONE_API_KEY)
idx = pc.Index(PINECONE_INDEX)
model = SentenceTransformer(MODEL_NAME)

app = FastAPI()

class QueryReq(BaseModel):
    question: str
    scope: str = "personal"   # personal|work|both
    top_k: int = 5

def check_auth(auth: Optional[str]):
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = auth.split(" ", 1)[1].strip()
    if token != API_BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid bearer token")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query")
def query(req: QueryReq, authorization: Optional[str] = Header(default=None)):
    check_auth(authorization)

    scope = req.scope.lower().strip()
    if scope == "personal":
        filt = {"scope": "personal"}
    elif scope == "work":
        filt = {"scope": "work"}
    elif scope == "both":
        filt = {"scope": {"$in": ["personal", "work"]}}
    else:
        raise HTTPException(status_code=400, detail="scope must be personal|work|both")

    qvec = model.encode([req.question], normalize_embeddings=True)[0].tolist()
    res = idx.query(vector=qvec, top_k=req.top_k, include_metadata=True, filter=filt)

    sources = []
    for m in res.get("matches", []) or []:
        md = m.get("metadata", {}) or {}
        sources.append({
            "title": md.get("title"),
            "deep_link": md.get("deep_link"),
            "source_system": md.get("source_system"),
            "scope": md.get("scope"),
            "modified_at": md.get("modified_at"),
            "score": m.get("score"),
            "snippet": (md.get("text","")[:400] + "...") if md.get("text") else None,
        })

    # Retrieval-only answer (LLM runs locally on your Mac, not on Render)
    answer = "Retrieved relevant sources from your knowledge base."

    return {"answer": answer, "scope_used": scope, "sources": sources}

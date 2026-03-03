import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

load_dotenv()

API_BEARER_TOKEN = os.environ.get("API_BEARER_TOKEN", "change_me")

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX = os.environ["PINECONE_INDEX"]
EMBED_MODEL_NAME = os.environ.get("LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

LLM_MODEL_PATH = os.environ.get("LLM_MODEL_PATH", "models/qwen2.5-1.5b-instruct.Q4_K_M.gguf")
LLM_CTX = int(os.environ.get("LLM_CTX", "4096"))

pc = Pinecone(api_key=PINECONE_API_KEY)
idx = pc.Index(PINECONE_INDEX)

embedder = SentenceTransformer(EMBED_MODEL_NAME)

llm = Llama(
    model_path=LLM_MODEL_PATH,
    n_ctx=LLM_CTX,
    # n_threads left default; llama-cpp will pick something sensible
)

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

def build_prompt(question: str, matches) -> str:
    # Keep context compact so it fits in ctx window
    context_blocks = []
    for m in matches:
        md = m.get("metadata", {}) or {}
        text = (md.get("text") or "").strip()
        title = (md.get("title") or "untitled").strip()
        if text:
            context_blocks.append(f"[{title}]\n{text}")

    context = "\n\n".join(context_blocks)[:12000]  # simple cap

    return (
        "You are a helpful assistant. Answer the user's question using ONLY the context provided. "
        "If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

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

    qvec = embedder.encode([req.question], normalize_embeddings=True)[0].tolist()
    res = idx.query(vector=qvec, top_k=req.top_k, include_metadata=True, filter=filt)

    matches = res.get("matches", []) or []

    sources = []
    for m in matches:
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

    prompt = build_prompt(req.question, matches)
    out = llm(prompt, max_tokens=300, temperature=0.2, stop=["\n\nContext:", "\nContext:", "\nQuestion:"])
    answer = (out["choices"][0]["text"] or "").strip()

    return {"answer": answer, "scope_used": scope, "sources": sources}

@app.get("/chat")
def chat():
    return FileResponse("static/chat.html")

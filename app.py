import os
import logging
from typing import Optional
from contextlib import asynccontextmanager
import asyncio

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_BEARER_TOKEN = os.environ.get("API_BEARER_TOKEN", "change_me")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX")
MODEL_NAME = os.environ.get("LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

pc = None
idx = None
model = None


def load_heavy_resources():
    global pc, idx, model

    logger.info("Initialising Pinecone...")
    try:
        if PINECONE_API_KEY and PINECONE_INDEX:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            idx = pc.Index(PINECONE_INDEX)
            logger.info("Pinecone initialised successfully.")
        else:
            logger.warning("PINECONE_API_KEY or PINECONE_INDEX is missing. Pinecone not initialised.")
    except Exception as e:
        logger.error(f"Failed to initialise Pinecone: {e}")

    logger.info("Loading SentenceTransformer...")
    try:
        model = SentenceTransformer(MODEL_NAME)
        logger.info(f"SentenceTransformer '{MODEL_NAME}' loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load SentenceTransformer: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Scheduling heavy resources initialisation in the background...")
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, load_heavy_resources)
    yield
    logger.info("Shutting down...")


app = FastAPI(lifespan=lifespan)


class QueryReq(BaseModel):
    question: str
    scope: str = "personal"  # personal|work|both
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

    if model is None or idx is None:
        raise HTTPException(
            status_code=503,
            detail="Search services (Model or VectorDB) are currently unavailable."
        )

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
            "snippet": (md.get("text", "")[:400] + "...") if md.get("text") else None,
        })

    return {"answer": "Retrieved relevant sources from your knowledge base.", "scope_used": scope, "sources": sources}

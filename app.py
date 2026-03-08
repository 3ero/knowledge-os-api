import os
import logging
from typing import Optional
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

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

def get_pinecone_indices():
    global pc, idx
    if pc is None or idx is None:
        logger.info("Initialising Pinecone...")
        try:
            from pinecone import Pinecone
            if PINECONE_API_KEY and PINECONE_INDEX:
                pc = Pinecone(api_key=PINECONE_API_KEY)
                idx = pc.Index(PINECONE_INDEX)
                logger.info("Pinecone initialised successfully.")
            else:
                logger.warning("PINECONE_API_KEY or PINECONE_INDEX is missing.")
        except Exception as e:
            logger.error(f"Failed to initialise Pinecone: {e}")
            raise HTTPException(status_code=503, detail=f"Pinecone init failed: {e}")
    return pc, idx

def get_embed_model():
    global model
    if model is None:
        logger.info("Loading SentenceTransformer...")
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(MODEL_NAME)
            logger.info(f"SentenceTransformer '{MODEL_NAME}' loaded.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(status_code=503, detail=f"Model load failed: {e}")
    return model

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    logger.info("Shutting down...")

app = FastAPI(lifespan=lifespan)

class QueryReq(BaseModel):
    question: str
    scope: str = "personal"
    top_k: int = 2

def check_auth(auth: Optional[str] = None):
    if not auth:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization format")
    token = auth.split(" ", 1)[1].strip()
    if token != API_BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid bearer token")

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query")
def query(req: QueryReq, authorization: Optional[str] = Header(default=None)):
    check_auth(authorization)
    _, current_idx = get_pinecone_indices()
    current_model = get_embed_model()
    if current_model is None or current_idx is None:
        raise HTTPException(status_code=503, detail="Services unavailable.")
    scope = req.scope.lower().strip()
    if scope == "personal":
        filt = {"scope": "personal"}
    elif scope == "work":
        filt = {"scope": "work"}
    elif scope == "both":
        filt = {"scope": {"$in": ["personal", "work"]}}
    else:
        raise HTTPException(status_code=400, detail="scope must be personal|work|both")
    actual_top_k = min(req.top_k, 2)
    qvec = current_model.encode([req.question], normalize_embeddings=True)[0].tolist()
    res = current_idx.query(vector=qvec, top_k=actual_top_k, include_metadata=True, filter=filt)
    sources = []
    for m in res.get("matches", []) or []:
        md = m.get("metadata", {}) or {}
        sources.append({
            "title": (md.get("title") or "")[:80],
            "snippet": (md.get("text") or "")[:100],
            "link": md.get("deep_link"),
        })
    return {"results": sources}

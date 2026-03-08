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
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL_NAME = os.environ.get("EMBED_MODEL", "text-embedding-3-small")

pc = None
idx = None
openai_client = None

def verify_and_migrate_pinecone_index(pc, index_name):
    import time
    try:
        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
        if index_name in existing_indexes:
            idx_info = pc.describe_index(index_name)
            current_dim = getattr(idx_info, 'dimension', idx_info.get('dimension', 0))
            if current_dim == 1536:
                logger.info(f"Pinecone index '{index_name}' already has dimension 1536. No migration needed.")
                return idx_info.spec

            logger.warning(f"Pinecone index '{index_name}' has dimension {current_dim}. Migrating to 1536...")
            old_spec = idx_info.spec
            pc.delete_index(index_name)
            
            while index_name in [info["name"] for info in pc.list_indexes()]:
                logger.info("Waiting for old index deletion...")
                time.sleep(2)
        else:
            from pinecone import ServerlessSpec
            old_spec = ServerlessSpec(cloud="aws", region="us-east-1")
            
        logger.info(f"Creating new Pinecone index '{index_name}' with dimension 1536...")
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=old_spec
        )
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(2)
        logger.info(f"Index '{index_name}' created successfully.")
    except Exception as e:
        logger.error(f"Failed to verify/migrate Pinecone index: {e}")

def get_pinecone_indices():
    global pc, idx
    if pc is None or idx is None:
        logger.info("Initialising Pinecone...")
        try:
            from pinecone import Pinecone
            if PINECONE_API_KEY and PINECONE_INDEX:
                pc = Pinecone(api_key=PINECONE_API_KEY)
                verify_and_migrate_pinecone_index(pc, PINECONE_INDEX)
                idx = pc.Index(PINECONE_INDEX)
                logger.info("Pinecone initialised successfully.")
            else:
                logger.warning("PINECONE_API_KEY or PINECONE_INDEX is missing.")
        except Exception as e:
            logger.error(f"Failed to initialise Pinecone: {e}")
            raise HTTPException(status_code=503, detail=f"Pinecone init failed: {e}")
    return pc, idx

def get_openai_client():
    global openai_client
    if openai_client is None:
        logger.info("Initializing OpenAI client...")
        try:
            from openai import OpenAI
            if OPENAI_API_KEY:
                openai_client = OpenAI(api_key=OPENAI_API_KEY)
                logger.info(f"OpenAI client loaded. Using model: {MODEL_NAME}")
            else:
                logger.warning("OPENAI_API_KEY is missing.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(status_code=503, detail=f"OpenAI load failed: {e}")
    return openai_client

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: pre-warming ML models and Vector API connections...")
    get_pinecone_indices()
    get_openai_client()
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
    current_openai = get_openai_client()
    if current_openai is None or current_idx is None:
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
    
    # Allow up to 5 results to give ChatGPT enough context without going crazy
    actual_top_k = min(req.top_k, 5) if req.top_k > 0 else 5
    
    try:
        embed_res = current_openai.embeddings.create(input=[req.question], model=MODEL_NAME)
        qvec = embed_res.data[0].embedding
    except Exception as e:
        logger.error(f"OpenAI Embedding failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate embedding: {e}")

    res = current_idx.query(vector=qvec, top_k=actual_top_k, include_metadata=True, filter=filt)
    sources = []
    for m in res.get("matches", []) or []:
        md = m.get("metadata", {}) or {}
        sources.append({
            "title": (md.get("title") or "")[:200],
            "snippet": (md.get("text") or "")[:1500],
            "link": md.get("deep_link"),
            "score": m.get("score"),
            "scope": md.get("scope"),
        })
    # Return both `sources` and `results` arrays to be absolutely sure we don't 
    # fail OpenAPI parameter validation in ChatGPT Custom Actions
    return {"results": sources, "sources": sources}

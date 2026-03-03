import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
idx = pc.Index(os.environ["PINECONE_INDEX"])

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

q = "What does my first knowledge OS document say?"
qvec = model.encode([q], normalize_embeddings=True)[0].tolist()

res = idx.query(vector=qvec, top_k=3, include_metadata=True, filter={"scope":"personal"})
print("Matches:")
for m in res["matches"]:
    md = m.get("metadata", {})
    print("-", md.get("title"), "| score:", m.get("score"))
    print("  source:", md.get("deep_link"))
    print("  snippet:", (md.get("text","")[:200] + "..."))

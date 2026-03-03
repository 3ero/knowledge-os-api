import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = os.environ["PINECONE_INDEX"]

idx = pc.Index(index_name)

# Upsert one dummy vector (dimension 1536)
vec = [0.0] * 1536
vec[0] = 1.0

idx.upsert(vectors=[{
    "id": "test:0",
    "values": vec,
    "metadata": {
        "scope": "personal",
        "source_system": "manual",
        "title": "pinecone smoke test",
        "deep_link": "local://pinecone_test",
        "modified_at": "2026-03-03T00:00:00Z"
    }
}])

res = idx.query(vector=vec, top_k=3, include_metadata=True, filter={"scope": "personal"})
print("Query results:")
for m in res["matches"]:
    print("-", m["id"], "score=", m["score"], "title=", m.get("metadata", {}).get("title"))

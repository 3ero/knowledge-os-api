# Knowledge OS API

The Knowledge OS API is a lightweight, high-performance vector retrieval API designed to integrate your personal knowledge base directly into ChatGPT Custom GPTs, automated workflows, and front-end applications.

It uses **FastAPI** for instant response times, **Pinecone** for serverless vector similarity search, and **OpenAI's** `text-embedding-3-small` model to generate high-quality 1536-dimensional semantic embeddings.

## Table of Contents

1. [Architecture](#architecture)
2. [Deployment](#deployment)
3. [Environment Configuration](#environment-configuration)
4. [API Endpoints](#api-endpoints)
5. [Local Ingestion Automation](#local-ingestion-automation)

## Architecture

- **Web Framework:** FastAPI + Uvicorn
- **Embeddings:** `openai` (`text-embedding-3-small`)
- **Vector Database:** `pinecone`
- **Hosting Target:** Render (Web Service)

The service is highly optimized to run on low-memory tiers (e.g., Render 512MB Starter) by lazy-loading API clients and avoiding heavy local ML dependencies like PyTorch.

## Deployment

This repository is pre-configured to automatically deploy on render.com.
If changes are pushed to `main`, Render will automatically build the service using `build.sh` and bind the application to `$PORT`.

### Automatic Database Migration

If you deploy this API to an existing Pinecone project that previously used 384-dimension models, the API will automatically perform a zero-downtime migration on startup:

1. It detects that the `PINECONE_INDEX` has 384 dimensions.
2. It safely deletes the old index.
3. It instantly creates a new index identical in configuration but with `1536` dimensions to perfectly match OpenAI embeddings.

## Environment Configuration

The following environment variables are required. On Render, set these in the **Environment** tab of the dashboard. Locally, place them in a `.env` file at the root of the project:

```env
PINECONE_API_KEY="your-pinecone-key"
PINECONE_INDEX="knowledge-os"
OPENAI_API_KEY="sk-proj-your-openai-api-key"
API_BEARER_TOKEN="your-secure-custom-token"  # Required for all API requests
EMBED_MODEL="text-embedding-3-small"         # Optional. Defaults to text-embedding-3-small
```

## API Endpoints

All endpoints (except `/health` and `/`) require an `Authorization` header containing your `API_BEARER_TOKEN`:
`Authorization: Bearer your-secure-custom-token`

### `GET /health`

A simple health check to ensure the server is alive and bound.

### `POST /query`

Performs a semantic similarity search against your knowledge base.

**Request:**

```json
{
  "question": "What is my favorite color?",
  "top_k": 5,                 // Optional. Max 5 results returned per query.
  "scope": "personal"         // Options: "personal", "work", or "both"
}
```

**Response:**

```json
{
  "results": [
    {
      "title": "My Diary",
      "snippet": "...my favorite color is ultra-violet...",
      "link": "https://url.to/source",
      "score": 0.892,
      "scope": "personal"
    }
  ],
  "sources": [ /* Identical to results to ensure ChatGPT OpenAPI compatibility */ ]
}
```

### `POST /ingest`

Allows you to directly upload and embed text documents from client applications or CI/CD pipelines directly into Pinecone from the cloud. The API handles chunking (1200 chars with 200 overlap), OpenAI embedding generation, and Pinecone upserting automatically.

**Request:**

```json
{
  "title": "Meeting Notes - Project X",
  "text": "Today we decided to migrate our entire stack to Pinecone and OpenAI...",
  "scope": "work",
  "source_system": "api",     // Let's you filter or identify how this document was added
  "deep_link": "https://notion.so/meeting-notes/123" // Optional callback link
}
```

**Response:**

```json
{
  "status": "ok",
  "doc_id": "838c8ca2a1fa11c7b33ec76b",
  "chunks_upserted": 2
}
```

## Local Ingestion Automation

If you prefer to organize text files locally on your machine and bulk-upload them to the cloud, use the included Python automation scripts.

### 1. `ingest_manual.py`

Place any `.txt` or `.md` files inside the `/data/manual/` directory. Simply run:

```bash
python3 ingest_manual.py
```

This script will recurse through the custom directory, embed the files using OpenAI, push them to Pinecone, and gracefully mark which ones have been processed so you do not double-bill your OpenAI tokens.

### 2. `watch_ingest.py`

A daemon process that watches the `/data/manual/` directory for any new files.

```bash
python3 watch_ingest.py
```

Leave this running in a terminal. Whenever you save a `.txt` or `.md` file into the `/data/manual` directory, it instantly routes it to `ingest_manual.py` for cloud ingestion without any manual commands.

#!/usr/bin/env bash
# exit on error
set -o errexit

pip install -r requirements.txt

# Download model during build-time so it's baked into the image
# This prevents OpenAI from hitting a 45-second timeout on the first request
python -c "from sentence_transformers import SentenceTransformer; import os; SentenceTransformer(os.environ.get('LOCAL_EMBED_MODEL', 'all-MiniLM-L6-v2'))"

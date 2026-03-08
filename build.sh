#!/usr/bin/env bash
# exit on error
set -o errexit

pip install -r requirements.txt

# Temporarily skipping model download step so the deploy succeeds, allowing the user to upgrade the instance.
# python -c "from sentence_transformers import SentenceTransformer; import os; SentenceTransformer(os.environ.get('LOCAL_EMBED_MODEL', 'all-MiniLM-L6-v2'))"

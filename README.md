Recipe RAG — quick start

This repository contains a small RAG pipeline (FAISS + BM25 hybrid search) and an LLM wrapper. The app is runnable as a CLI that builds embeddings/indexes and accepts queries interactively.

Quick local steps

1. Create virtualenv and install requirements

```ps1
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r src/requirements.txt
```

2. Make sure you have the dataset file at `src/data/raw/recipes.json` (see `src/config/config.py` DATA_PATH).

3. Run the app

```ps1
python src/main.py
```

Run with Docker

1. Build

```ps1
docker build -t recipes_rag:latest .
```

2. Run

```ps1
docker run --rm -it -v ${PWD}/src/data:/app/data recipes_rag:latest
```

Or use docker-compose:

```ps1
docker compose up --build
```

Notes and caveats

- You must provide a dataset JSON file described in `src/config/config.py`.
- Some LLM models referenced in `config/config.py` may be large and require auth + GPU. If you don't have GPU or HF access, switch `LLM_MODEL` to a small local model or mock.
- FAISS and heavy ML libs increase image size. For experimentation prefer running in a Python venv.

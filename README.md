# RCA RAG MVP

Simple and fast RCA platform built as a single FastAPI app with:

- one upload endpoint
- one analyze endpoint
- one similarity endpoint
- one incident list endpoint
- a plain HTML/CSS/JS UI
- FAISS search with NumPy fallback
- template RCA fallback when Llama3 or OpenAI is unavailable

## Project Structure

```text
app/
├── main.py
├── config.py
├── models.py
├── services/
│   ├── ingest_service.py
│   ├── embedding_service.py
│   ├── retrieval_service.py
│   ├── rca_service.py
│   └── vector_store.py
├── utils/
│   ├── preprocessing.py
│   └── logger.py
├── static/
│   ├── index.html
│   ├── app.js
│   └── styles.css
└── data/
    ├── incidents.json
    └── faiss_index/
```

## Run Locally

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://localhost:8000/`.

## Llama3 Setup

If you want local Llama3 RCA generation through Ollama:

```bash
export LLM_PROVIDER=llama3
export LLAMA3_MODEL=llama3.1
export LLAMA3_BASE_URL=http://localhost:11434/api/generate

ollama serve
ollama pull llama3.1
uvicorn app.main:app --reload
```

If Ollama is unavailable, the app falls back automatically to template RCA using retrieved incidents.

## API

### `GET /`

Serves the single-page UI.

### `GET /health`

Returns app status, incident count, index readiness, embedding backend, and LLM provider.

### `POST /ingest`

Upload `.json` or `.txt` incident logs.

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@sample_logs/logs_1.json"
```

### `POST /analyze`

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"log":"ERROR: Redis cache connection reset by peer in order-service"}'
```

Example response:

```json
{
  "root_cause": "Likely caused by redis cache connection reset by peer during order lookup similar to historical incident INC1003.",
  "severity": "HIGH",
  "suggested_fix": "Restarted Redis primary, flushed the broken connection pool, and raised client pool limits",
  "confidence_score": 0.89,
  "similar_incidents": [
    {
      "incident_id": "INC1003",
      "service_name": "order-service",
      "similarity": 0.89,
      "resolution": "Restarted Redis primary, flushed the broken connection pool, and raised client pool limits",
      "severity": "HIGH",
      "error_message": "Redis cache connection reset by peer during order lookup"
    }
  ]
}
```

### `GET /incidents`

Returns stored incidents.

### `GET /similar`

```bash
curl "http://localhost:8000/similar?message=database%20timeout&top_k=5"
```

## Testing

```bash
pytest
```

## Docker

Optional container run:

```bash
docker compose up --build
```

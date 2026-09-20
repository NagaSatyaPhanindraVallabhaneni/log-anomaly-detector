# Log Anomaly Detector

A real-time log anomaly detection service. It ingests application log lines, learns what "normal" looks like from a baseline, and scores incoming logs as normal or anomalous — surfacing error bursts, latency spikes, and never-before-seen log templates.

Built as a portfolio project to demonstrate production-style Python: a typed core library, a FastAPI service, synthetic data generation for reproducible demos, a real test suite, Docker packaging, and CI.

## Architecture

```
                    +------------------+
                    | synthetic logs   |  scripts/generate_logs.py
                    | (seeded, no      |  --mode normal|mixed
                    |  external data)  |
                    +--------+---------+
                             |
                             v
 +--------+   +------------------+   +-------------------+   +----------------+
 | client +-->+  FastAPI service  +-->+ parser + template  +-->+ IsolationForest|
 | (curl) |   | /train /ingest   |   | miner + 16 numeric |   | (scikit-learn) |
 +--------+   | /score /stats    |   | features           |   | score =        |
              | /health          |   +-------------------+   | -decision_fn   |
              +--------+---------+                             +-------+--------+
                       |                                             |
                       v                                             v
              rolling buffer + stats                        joblib model file
              (in-memory)                                   (models/detector.joblib)
```

## How the model works

1. **Parse** — each line is split into timestamp, level, service, message, and `key=value` attributes. Malformed lines never crash the pipeline; they are parsed with `level="UNKNOWN"`.
2. **Template mining** — volatile tokens (numbers, IPs, UUIDs, hex, quoted strings) are masked so `login failed user_id=4821` and `login failed user_id=9917` become one template. Rarity of each template is computed against the training baseline; a never-seen template gets maximum rarity.
3. **Features** — 16 numeric features per line: level one-hots, log-latency, status-code buckets, cyclic hour-of-day, message length / token count / digit ratio, and template rarity.
4. **Model** — features are standardized and fed to an `IsolationForest` trained on the baseline. Anomaly score is `-decision_function(x)` (higher = more anomalous); the binary flag comes from the forest's own prediction.

## Quickstart

### One-command run (Docker)

```bash
docker compose up --build
# service on http://localhost:8000
```

### Local run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn log_anomaly_detector.service:app --reload   # needs PYTHONPATH=src
# or: PYTHONPATH=src uvicorn log_anomaly_detector.service:app --reload
```

### CLI demo (no server needed)

Trains on 2,000 generated normal logs, scores a mixed stream with injected anomalies, and prints the top anomalies plus recall/precision against the synthetic ground truth:

```bash
PYTHONPATH=src python scripts/demo.py
```

Generate logs yourself:

```bash
python scripts/generate_logs.py --n 1000 --mode mixed --seed 42 --out logs.jsonl
```

## API reference

Train on a baseline of normal logs:

```bash
curl -s -X POST localhost:8000/train \
  -H 'Content-Type: application/json' \
  -d @- <<'EOF' | head -c 300
{"lines": ["2026-09-20T04:00:01.123Z INFO auth login successful user_id=4821 latency_ms=42 status=200"]}
EOF
```

Score lines without storing them (`/ingest` scores *and* updates stats):

```bash
curl -s -X POST localhost:8000/score \
  -H 'Content-Type: application/json' \
  -d '{"lines": ["2026-09-20T04:05:11.000Z ERROR payments database connection timeout after 5000ms latency_ms=5000 status=500"]}'
```

```json
[{"raw": "...", "service": "payments", "level": "ERROR",
  "template": "database connection timeout after <NUM>ms",
  "score": 12.4, "is_anomaly": true}]
```

Health and stats:

```bash
curl -s localhost:8000/health
curl -s localhost:8000/stats
```

| Endpoint      | Method | Description                                              |
|---------------|--------|----------------------------------------------------------|
| `/train`      | POST   | Train on baseline lines, persist model to disk           |
| `/score`      | POST   | Score lines (stateless)                                  |
| `/ingest`     | POST   | Score lines and update rolling stats                     |
| `/health`     | GET    | Liveness + whether a model is loaded                     |
| `/stats`      | GET    | Ingested/anomaly counts, anomaly rate, templates known   |

`POST /train` accepts an optional `contamination` (expected anomaly fraction, default `0.02`). The model file is reloaded automatically on restart.

## Tests & CI

```bash
pytest -q          # full suite: parser, features, model, API
ruff check src scripts tests
```

GitHub Actions runs ruff + pytest on every push and pull request.

## Tech stack

Python 3.10+ · FastAPI / Uvicorn · scikit-learn (IsolationForest) · pandas / NumPy · Pydantic · joblib · pytest · httpx · Ruff · Docker / Compose · GitHub Actions

## Notes

- All demo data is synthetic and seeded — nothing here claims real production traffic, users, or deployments.
- Anomaly scores are relative to the baseline you train on; retrain when your log patterns change.

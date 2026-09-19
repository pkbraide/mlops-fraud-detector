---
title: Enterprise Real-Time Fraud Detection Engine
emoji: 🛡️
colorFrom: red
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
license: mit
short_description: Real-time card fraud scoring API (FastAPI + scikit-learn)
---

# Enterprise Real-Time Fraud Detection Engine

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/pkbraide/mlops-fraud-detector)

[![CI/CD](https://github.com/pkbraide/mlops-fraud-detector/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/pkbraide/mlops-fraud-detector/actions/workflows/ci-cd.yml)
![Python](https://img.shields.io/badge/python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-e92063)
![Docker](https://img.shields.io/badge/docker-python%3A3.10--slim-2496ED)
![License](https://img.shields.io/badge/license-MIT-green)

A production-shaped, low-latency REST service that scores card transactions for
fraud in real time. A scikit-learn `RandomForestClassifier` is trained in a
reproducible notebook, serialised with `joblib`, and served by FastAPI behind a
strict Pydantic v2 request contract. The service ships with a deterministic
rule-based fallback, a hardened non-root Docker image, a GitHub Actions
pipeline that lints, tests and smoke-tests the container, and one-click
deployment blueprints for Render and Hugging Face Spaces.

**Live URL:** <https://mlops-fraud-detector.onrender.com> · Interactive Swagger UI: <https://mlops-fraud-detector.onrender.com/docs>

```bash
curl -s https://mlops-fraud-detector.onrender.com/
# {"status":"healthy","service":"Enterprise Real-Time Fraud Detection Engine","model_version":"joblib_model","mock_mode":false}
```

> Hosted on Render's free tier, which spins down after 15 minutes of inactivity —
> the first request after idle can take 30–60 s while the container cold-starts.

---

## Table of contents

1. [Architecture](#architecture)
2. [Quick start](#quick-start)
3. [API reference](#api-reference)
4. [Fallback Mock Model](#fallback-mock-model)
5. [Model training](#model-training)
6. [Testing & code quality](#testing--code-quality)
7. [CI/CD](#cicd)
8. [Deployment](#deployment)
9. [Project layout](#project-layout)
10. [Configuration](#configuration)

---

## Architecture

```
                         ┌──────────────────────────────────────────────┐
                         │              notebooks/train.ipynb           │
   offline / batch       │  make_classification ─► feature scaling      │
                         │  ─► RandomForest(100 trees) ─► joblib.dump   │
                         └───────────────────────┬──────────────────────┘
                                                 │ models/fraud_model.joblib
                                                 │ (committed to git, baked into image)
 ────────────────────────────────────────────────┼─────────────────────────────────────────
   online / real-time                            ▼
                     ┌──────────────────────────────────────────────────────────┐
   HTTPS  ┌───────┐  │  Docker: python:3.10-slim, non-root `appuser`            │
 ────────►│ Load  │  │  ┌────────────────────────────────────────────────────┐  │
  client  │Balancer│─►│  │  uvicorn ─► FastAPI  (app/main.py)                 │  │
 ◄────────│(Render │  │  │   GET  /          health + model_version           │  │
   JSON   │ / HF)  │  │  │   POST /predict   Pydantic v2 validation ──┐       │  │
          └───────┘  │  │                                              ▼       │  │
                     │  │  FraudDetectionModel singleton (app/model_utils.py)  │  │
                     │  │   ├─ joblib artifact loaded once at import           │  │
                     │  │   │    └─ model_version = "joblib_model"             │  │
                     │  │   └─ fallback rule when artifact is missing/corrupt  │  │
                     │  │        └─ model_version = "mock_rule"                │  │
                     │  └────────────────────────────────────────────────────┘  │
                     └──────────────────────────────────────────────────────────┘
                                                 ▲
                     ┌───────────────────────────┴──────────────────────────┐
                     │  GitHub Actions (.github/workflows/ci-cd.yml)        │
                     │  black --check ─► flake8 ─► pytest ─► docker build   │
                     │  ─► container smoke test ─► Render auto-deploy       │
                     └──────────────────────────────────────────────────────┘
```

**Design decisions**

| Concern | Decision | Why |
|---|---|---|
| Model loading | Module-level singleton, loaded once at import | Zero per-request I/O; cold-start cost paid exactly once per worker. |
| Input contract | Pydantic v2 `Field` constraints (`gt=0`, `ge=0, le=1`) | Malformed payloads are rejected with a structured 422 before touching the model. |
| Resilience | Rule-based fallback when the artifact is absent | The API contract is stable across CI, fresh clones and partial deploys. |
| Observability | `model_version` on every response and on `/` | Consumers and dashboards can tell which decision path served a request. |
| Security | Non-root container user, slim base image, no build tools at runtime | Smaller attack surface, passes common container-hardening checks. |
| Reproducibility | Pinned `requirements.txt`, `random_state=42` everywhere | Same artifact, same predictions, on any machine. |

---

## Quick start

### Prerequisites

- Python **3.10** (the version the model artifact and pins are validated against)
- `git`, `curl`
- Docker (optional, for the container workflow)

### Local development

```bash
git clone https://github.com/pkbraide/mlops-fraud-detector.git
cd mlops-fraud-detector

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# Start the API with hot reload
uvicorn app.main:app --reload --port 8000
```

Interactive docs are served at <http://localhost:8000/docs> (Swagger UI) and
<http://localhost:8000/redoc>.

### Run with Docker

```bash
docker build -t mlops-fraud-detector .
docker run --rm -p 8000:8000 mlops-fraud-detector

# Platforms that inject $PORT (Render, Railway, HF Spaces) are honoured:
docker run --rm -e PORT=7860 -p 7860:7860 mlops-fraud-detector
```

### Run the tests

```bash
pytest -v
```

---

## API reference

| Method | Path | Description | Success | Errors |
|---|---|---|---|---|
| `GET` | `/` | Health probe; reports service name, `model_version` and `mock_mode`. | `200` | — |
| `POST` | `/predict` | Score one transaction. | `200` | `422` validation, `500` inference failure |
| `GET` | `/docs` | Swagger UI. | `200` | — |
| `GET` | `/openapi.json` | OpenAPI 3.1 schema. | `200` | — |

### `POST /predict`

**Request body**

| Field | Type | Constraint | Meaning |
|---|---|---|---|
| `amount` | `float` | `> 0` | Transaction amount in USD |
| `distance_from_home` | `float` | `> 0` | Distance in km between merchant and cardholder's home |
| `use_chip` | `int` | `0` or `1` | `1` if the EMV chip was used, `0` for swipe / manual entry |

**Response body**

| Field | Type | Meaning |
|---|---|---|
| `prediction` | `int` | `1` = fraud, `0` = legitimate |
| `is_fraud` | `bool` | Convenience boolean of `prediction` |
| `model_version` | `str` | `"joblib_model"` or `"mock_rule"` — see [Fallback Mock Model](#fallback-mock-model) |

**Example**

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"amount": 2500, "distance_from_home": 180.0, "use_chip": 0}'
```

```json
{"prediction": 1, "is_fraud": true, "model_version": "joblib_model"}
```

> This payload is flagged by **both** decision paths: the trained forest and
> the fallback rule (`amount > 1000`, `distance_from_home > 50`, `use_chip == 0`),
> so the example is stable whichever `model_version` is serving.

**Validation failure (422)**

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"amount": "not_a_number", "distance_from_home": 10.0, "use_chip": 1}'
```

```json
{
  "detail": [
    {
      "type": "float_parsing",
      "loc": ["body", "amount"],
      "msg": "Input should be a valid number, unable to parse string as a number",
      "input": "not_a_number"
    }
  ]
}
```

### `GET /`

```bash
curl -s http://localhost:8000/
```

```json
{
  "status": "healthy",
  "service": "Enterprise Real-Time Fraud Detection Engine",
  "model_version": "joblib_model",
  "mock_mode": false
}
```

---

## Fallback Mock Model

`app/model_utils.py` wraps the classifier in `FraudDetectionModel`. On
construction it calls `joblib.load("models/fraud_model.joblib")`:

- **Artifact present and loadable** → `model_version = "joblib_model"`,
  `is_mock = False`, and `/predict` runs the RandomForest.
- **Artifact missing, corrupt, or built with an incompatible scikit-learn** →
  a warning is logged, `model_version = "mock_rule"`, `is_mock = True`, and
  `/predict` evaluates a deterministic heuristic:

  ```python
  is_fraud = amount > 1000 and distance_from_home > 50.0 and use_chip == 0
  ```

**Why this matters**

- **CI stays green on day one.** A brand-new clone, a shallow checkout, or a
  runner that never trained the model can still import the app, serve
  requests, and pass the full test suite. Tests that assert a specific decision
  boundary are guarded on `fraud_model.is_mock`, so the same suite validates
  both paths.
- **No silent degradation.** Every response carries `model_version`, and the
  health endpoint carries `mock_mode`. A dashboard, an integration test, or an
  on-call engineer can see at a glance whether a request was scored by the
  trained model or by the fallback rule — the service never pretends to be
  something it is not.
- **Fail-safe, not fail-open.** The fallback still flags the highest-risk
  pattern (large, far from home, no chip) rather than returning `0` for
  everything.

`model_version` is constrained to exactly two literal values —
`"joblib_model"` and `"mock_rule"` — so it is safe to branch on.

---

## Model training

The model is trained in [`notebooks/train.ipynb`](notebooks/train.ipynb):

1. **Synthetic data.** `make_classification(n_samples=5000, n_features=3,
   n_informative=3, n_redundant=0, weights=[0.95, 0.05], random_state=42)`.
   Real transaction data is regulated (PCI-DSS / GDPR) and cannot live in a
   public repository; a synthetic set is licence-free and perfectly
   reproducible.
2. **Feature engineering.** Columns are named `amount`,
   `distance_from_home`, `use_chip` — in that order, which is the contract
   the API honours — then rescaled to realistic ranges (0–5000 USD,
   0–200 km) and `use_chip` is binarised.
3. **Class imbalance.** 95/5 mirrors the rare-event nature of fraud closely
   enough that accuracy is misleading and per-class precision/recall must be
   inspected, while still leaving ~250 positives to learn from.
4. **Model.** `RandomForestClassifier(n_estimators=100, random_state=42)`
   with a stratified 80/20 split and a `classification_report` on the
   hold-out set. Last recorded run (1 000 hold-out rows, 54 positives):

   | class | precision | recall | f1 | support |
   |---|---|---|---|---|
   | legit | 0.952 | 0.985 | 0.968 | 946 |
   | fraud | 0.333 | 0.130 | 0.187 | 54 |

   Fraud recall is deliberately not tuned: the features are synthetic and
   `use_chip` is binarised, so there is limited signal to recover. The point
   of this repository is the serving, testing and delivery pipeline, and the
   numbers above are reported as-is rather than cherry-picked.
5. **Artifact.** `joblib.dump(model, "../models/fraud_model.joblib")`
   followed by a reload round-trip check.

Re-train from the command line:

```bash
pip install jupyter nbconvert
jupyter nbconvert --to notebook --execute notebooks/train.ipynb --inplace
```

The artifact is **committed to git on purpose** so that the API serves the real
model on a fresh clone and inside the Docker image without a training step.

---

## Testing & code quality

```bash
black --check .   # formatting
flake8 .          # linting (config in setup.cfg)
pytest -v         # 13 tests across health, inference, validation and OpenAPI
```

`setup.cfg` pins `max-line-length = 88` and ignores `E203`/`W503` so black and
flake8 agree — without it, the two tools contradict each other and CI fails on
the very first push.

Test coverage includes:

| Test | Asserts |
|---|---|
| `test_health_check` | `200`, `status == "healthy"`, `model_version` is one of the two allowed literals |
| `test_predict_valid_legit` | `200`, all response keys present, `is_fraud` mirrors `prediction` |
| `test_predict_mock_fraud_trigger` | `200`; `prediction == 1` **only when** the mock is active |
| `test_predict_response_is_consistent` | Identical payloads yield identical decisions |
| `test_validation_string_in_float_field` | `422` for `"amount": "not_a_number"` |
| `test_validation_negative_amount` / `_zero_amount` | `422` for `amount <= 0` |
| `test_validation_bad_use_chip` | `422` for `use_chip = 5` |
| `test_validation_missing_field` (×3) | `422` and the missing field named in `loc` |
| `test_validation_empty_body` | `422` |
| `test_openapi_schema_exposes_example` | Pydantic v2 `json_schema_extra` example surfaces in OpenAPI |

---

## CI/CD

Workflow: [`.github/workflows/ci-cd.yml`](.github/workflows/ci-cd.yml)

Triggers on every `push` and `pull_request` targeting `main`. A single
`build-and-test` job on `ubuntu-latest` with Python 3.10 and a pip cache runs:

```
checkout ─► setup-python ─► pip install ─► black --check . ─► flake8 . ─► pytest -v
        ─► docker build ─► run container ─► curl / and /predict ─► assert joblib_model
```

The container smoke test asserts that the **real** model (`"joblib_model"`)
is what gets baked into the image, so a regression that drops the artifact
from the build context fails CI rather than shipping the fallback to
production. Concurrent runs on the same ref are cancelled to save minutes.

Continuous deployment is handled by the hosting platform: Render's
`autoDeploy: true` redeploys on every push to `main` once the blueprint is
connected, and Hugging Face Spaces rebuilds on every push to the Space remote.

---

## Deployment

### Render (primary) — Blueprint

`render.yaml` at the repository root declares a single Docker web service
(`plan: free`, `region: frankfurt`, `healthCheckPath: /`, `autoDeploy: true`).

1. Sign in at <https://dashboard.render.com> and connect your GitHub account.
2. **New → Blueprint**, select `mlops-fraud-detector`, click **Apply**.
3. Render builds the Dockerfile, injects `$PORT`, and the shell-form `CMD`
   picks it up. The health check on `/` must return `200` before traffic is
   routed.
4. Subsequent pushes to `main` redeploy automatically.

**Automated redeploys from the terminal.** After the first Blueprint apply,
copy the service's *Deploy Hook* URL (Settings → Deploy Hook) and export it:

```bash
export RENDER_DEPLOY_HOOK_URL="https://api.render.com/deploy/srv-xxxxxxxxxxxx?key=yyyyyyyy"
curl -X POST "$RENDER_DEPLOY_HOOK_URL"
```

> Free-tier Render services spin down after 15 minutes of inactivity; the
> first request after idle can take ~30–60 s.

### Hugging Face Spaces — always-on demo

The YAML frontmatter at the top of this README makes the repository a valid
Docker Space (`sdk: docker`, `app_port: 8000`). Spaces on the free CPU tier do
not sleep on inactivity the way Render's free tier does, so it is the better
target for a portfolio demo with no cold start.

```bash
# one-time
pip install huggingface_hub
huggingface-cli login
huggingface-cli repo create mlops-fraud-detector --type space --space_sdk docker

# push the same repo to the Space remote
git remote add hf https://huggingface.co/spaces/<your-hf-username>/mlops-fraud-detector
git push hf main
```

The Space exposes the API at
`https://<your-hf-username>-mlops-fraud-detector.hf.space` with Swagger UI at
`/docs`.

### Railway (alternative)

```bash
railway login
railway init
railway up
railway domain
```

Railway detects the Dockerfile and injects `$PORT` automatically.

---

## Project layout

```
mlops-fraud-detector/
├── .github/workflows/ci-cd.yml   # lint → test → docker build → smoke test
├── app/
│   ├── __init__.py
│   ├── main.py                   # FastAPI app, Pydantic v2 schemas, routes
│   └── model_utils.py            # FraudDetectionModel + module-level singleton
├── models/
│   ├── .gitkeep
│   └── fraud_model.joblib        # trained RandomForest (tracked in git)
├── notebooks/train.ipynb         # reproducible training pipeline
├── tests/
│   ├── __init__.py
│   └── test_api.py               # API contract tests (mock-aware)
├── Dockerfile                    # python:3.10-slim, non-root, cached deps layer
├── .dockerignore
├── .gitignore
├── setup.cfg                     # flake8 config aligned with black
├── render.yaml                   # Render Blueprint
├── requirements.txt              # exact pins, resolved for Python 3.10
└── README.md                     # this file (doubles as HF Space card)
```

---

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `PORT` | `8000` | Port uvicorn binds to. Injected by Render / Railway / HF Spaces. |
| `LOG_LEVEL` | `INFO` | Python logging level for the app and model loader. |

---

## License

MIT — see the badge above. Synthetic data only; no real cardholder data is
used or stored anywhere in this project.

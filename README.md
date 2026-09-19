---
title: Enterprise Real-Time Fraud Detection Engine
emoji: 🛡️
colorFrom: red
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
license: mit
short_description: Real-time card fraud scoring API + UI (FastAPI, scikit-learn)
---

# Enterprise Real-Time Fraud Detection Engine

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/pkbraide/mlops-fraud-detector)

[![CI/CD](https://github.com/pkbraide/mlops-fraud-detector/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/pkbraide/mlops-fraud-detector/actions/workflows/ci-cd.yml)
![Python](https://img.shields.io/badge/python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-e92063)
![Docker](https://img.shields.io/badge/docker-python%3A3.10--slim-2496ED)
![License](https://img.shields.io/badge/license-MIT-green)

A production-shaped, low-latency service that scores card transactions for
fraud in real time. A class-weighted scikit-learn `RandomForestClassifier` is
trained on **one million real, labelled transactions** (OpenML `card_transdata`,
CC0), serialised with `joblib`, and served by FastAPI behind a strict Pydantic v2
contract. A dependency-free browser UI at `/` lets a visitor score a transaction
in one click; `/v2/predict` is the JSON API behind it. The service ships with a
deterministic rule-based fallback, a hardened non-root Docker image, a GitHub
Actions pipeline that lints, tests, builds the container and smoke-tests it, and
one-click deployment blueprints for Render and Hugging Face Spaces.

**Live URL:** <https://mlops-fraud-detector.onrender.com> · Interactive Swagger UI: <https://mlops-fraud-detector.onrender.com/docs>

```bash
curl -s https://mlops-fraud-detector.onrender.com/health
# {"status":"healthy","service":"Enterprise Real-Time Fraud Detection Engine","version":"2.0.0","model_version":"joblib_model_v2","mock_mode":false,"endpoints":[...]}
```

> Hosted on Render's free tier, which spins down after 15 minutes of inactivity —
> the first request after idle can take 30–60 s while the container cold-starts.
> The UI says so on the page rather than leaving a visitor to assume it is broken.

**Screenshot:** _UI screenshot to be added at `docs/ui-screenshot.png` (capture from the live URL)._

---

## Table of contents

1. [Architecture](#architecture)
2. [Quick start](#quick-start)
3. [Web UI](#web-ui)
4. [API reference](#api-reference)
5. [v1 (deprecated) vs v2 (current)](#v1-deprecated-vs-v2-current)
6. [Fallback Mock Model](#fallback-mock-model)
7. [Model training & evaluation](#model-training--evaluation)
8. [Limitations](#limitations)
9. [Testing & code quality](#testing--code-quality)
10. [CI/CD](#cicd)
11. [Deployment](#deployment)
12. [Project layout](#project-layout)
13. [Configuration](#configuration)

---

## Architecture

```
                    ┌────────────────────────────────────────────────────────────┐
  offline / batch   │  notebooks/train.ipynb                                     │
                    │  fetch_openml(45955) ─► 1,000,000 rows ─► stratified split │
                    │  ─► RandomForest(class_weight="balanced") ─► joblib.dump   │
                    │     {"model": rf, "features": [...7 names...], "metrics"}  │
                    └───────────────────────────────┬────────────────────────────┘
                                                    │ models/fraud_model_v2.joblib (1.1 MB)
                                                    │ models/fraud_model.joblib    (v1, kept)
 ───────────────────────────────────────────────────┼───────────────────────────────────────
  online / real-time                                ▼
                 ┌──────────────────────────────────────────────────────────────────────┐
  browser ──────►│  Docker: python:3.10-slim, non-root `appuser`, uvicorn ─► FastAPI   │
  or curl        │                                                                      │
                 │   GET  /            Jinja2 page + /static/{style.css,app.js}         │
                 │   GET  /health      JSON: version, model_version, mock_mode, routes  │
                 │   POST /v2/predict  7 features  ─┐                                   │
                 │   POST /predict     3 features   │ (deprecated, byte-identical)      │
                 │                                  ▼                                   │
                 │   FraudDetectionModel singleton (app/model_utils.py)                 │
                 │     v2 artifact ─► "joblib_model_v2"   feature order read FROM the   │
                 │     v1 artifact ─► "joblib_model"      artifact, asserted at load    │
                 │     neither     ─► "mock_rule"         deterministic fallback rules  │
                 └──────────────────────────────────────────────────────────────────────┘
                                                    ▲
                 ┌──────────────────────────────────┴───────────────────────────────────┐
                 │  GitHub Actions: black ─► flake8 ─► pytest (34) ─► docker build      │
                 │  ─► assert static/templates/models inside image ─► run container     │
                 │  ─► curl / (HTML), /health, /v2/predict, /predict ─► Render deploys  │
                 └──────────────────────────────────────────────────────────────────────┘
```

**Design decisions**

| Concern | Decision | Why |
|---|---|---|
| Column alignment | v2 artifact is a dict carrying its own `features` list; the loader rejects it if the order differs from the API's | Feature order can never silently drift between training and inference. |
| Model loading | Module-level singleton, loaded once at import | Zero per-request I/O; cold-start cost paid once per worker. |
| Input contract | Pydantic v2 `Field` constraints (`ge=0`, `ge=0, le=1`) | Malformed payloads are rejected with a structured 422 before touching the model. |
| Versioning | `/predict` kept byte-identical and marked `deprecated=True`; `/v2/predict` added | Existing clients keep working; OpenAPI and `/health` both advertise the split. |
| Resilience | Rule-based fallback per endpoint when its artifact is absent | API contract stable across CI, fresh clones and partial deploys. |
| Observability | `model_version` on every response and per-route on `/health` | Dashboards and humans can see which decision path served a request. |
| Frontend | Vanilla HTML/CSS/JS served by the same process; no CDN, no build step | Works offline of npm, ships in the same image, one fewer moving part. |
| Security | Non-root container, slim base, no build tools at runtime | Smaller attack surface. |
| Reproducibility | Exact pins for Python 3.10, `random_state=42`, CC0 dataset fetched by id | Same artifact, same predictions, on any machine. |

---

## Quick start

### Prerequisites

- Python **3.10** (the version the artifacts and pins are validated against)
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

uvicorn app.main:app --reload --port 8000
```

Open <http://localhost:8000> for the UI, <http://localhost:8000/docs> for Swagger UI.

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

## Web UI

`GET /` serves a single-page UI (`app/templates/index.html`,
`app/static/style.css`, `app/static/app.js`). No framework, no bundler, no CDN.

- Header with the project title and a **live model badge** read from `/health`
  on page load: green for `joblib_model_v2`, amber for `mock_rule`, slate for
  `joblib_model`, red if the health probe is unreachable.
- A form covering all seven v2 features. Binary features are selects, distances
  and the ratio are numeric inputs; client-side validation mirrors the Pydantic
  constraints (`≥ 0`, binary `∈ {0, 1}`) and 422 responses are mapped back onto
  the offending fields.
- **Load legitimate example** / **Load fraudulent example** buttons that fill
  the form with a transaction the model classifies each way, so a visitor sees
  both outcomes in two clicks.
- Submit posts to `/v2/predict` and renders a verdict card — **FRAUD** (red) or
  **LEGITIMATE** (green) — with the `model_version` that served it, the round-trip
  latency and the raw JSON.
- Explicit loading and error states, a cold-start notice for the free tier,
  and a responsive layout verified at 390 px with no horizontal overflow.

---

## API reference

| Method | Path | Status | Description | Success | Errors |
|---|---|---|---|---|---|
| `GET` | `/` | current | Browser UI (HTML). Not in the OpenAPI schema. | `200 text/html` | — |
| `GET` | `/health` | current | JSON probe: `version`, `model_version`, `mock_mode`, per-route status. | `200` | — |
| `POST` | `/v2/predict` | **current** | Score a transaction with the real-data model (7 features). | `200` | `422`, `500` |
| `POST` | `/predict` | **deprecated** | v1 contract (3 synthetic features). Byte-identical to 1.x. | `200` | `422`, `500` |
| `GET` | `/docs` | current | Swagger UI. | `200` | — |
| `GET` | `/openapi.json` | current | OpenAPI 3.1 schema (`/predict` carries `deprecated: true`). | `200` | — |

### `POST /v2/predict`

**Request body**

| Field | Type | Constraint | Meaning |
|---|---|---|---|
| `distance_from_home` | `float` | `≥ 0` | km between transaction location and cardholder's home |
| `distance_from_last_transaction` | `float` | `≥ 0` | km from the previous transaction |
| `ratio_to_median_purchase_price` | `float` | `≥ 0` | transaction price ÷ cardholder's median purchase price |
| `repeat_retailer` | `int` | `0` or `1` | retailer previously used by this card |
| `used_chip` | `int` | `0` or `1` | EMV chip used |
| `used_pin_number` | `int` | `0` or `1` | PIN entered |
| `online_order` | `int` | `0` or `1` | card-not-present online purchase |

**Response body** (shared with v1)

| Field | Type | Meaning |
|---|---|---|
| `prediction` | `int` | `1` = fraud, `0` = legitimate |
| `is_fraud` | `bool` | Convenience boolean of `prediction` |
| `model_version` | `str` | `"joblib_model_v2"` or `"mock_rule"` for this route |

**Examples**

```bash
# Legitimate: small, nearby, in-person chip purchase at a known retailer
curl -s -X POST https://mlops-fraud-detector.onrender.com/v2/predict \
  -H "Content-Type: application/json" \
  -d '{"distance_from_home": 5.2, "distance_from_last_transaction": 0.4,
       "ratio_to_median_purchase_price": 0.9, "repeat_retailer": 1,
       "used_chip": 1, "used_pin_number": 0, "online_order": 0}'
# {"prediction":0,"is_fraud":false,"model_version":"joblib_model_v2"}

# Fraudulent: far from home, 6× the usual spend, online, no chip, no PIN, new retailer
curl -s -X POST https://mlops-fraud-detector.onrender.com/v2/predict \
  -H "Content-Type: application/json" \
  -d '{"distance_from_home": 210.5, "distance_from_last_transaction": 12.3,
       "ratio_to_median_purchase_price": 6.2, "repeat_retailer": 0,
       "used_chip": 0, "used_pin_number": 0, "online_order": 1}'
# {"prediction":1,"is_fraud":true,"model_version":"joblib_model_v2"}

# Validation failure: binary field out of range
curl -s -X POST https://mlops-fraud-detector.onrender.com/v2/predict \
  -H "Content-Type: application/json" \
  -d '{"distance_from_home": 5.2, "distance_from_last_transaction": 0.4,
       "ratio_to_median_purchase_price": 0.9, "repeat_retailer": 1,
       "used_chip": 2, "used_pin_number": 0, "online_order": 0}'
# HTTP 422
# {"detail":[{"type":"less_than_equal","loc":["body","used_chip"],
#             "msg":"Input should be less than or equal to 1","input":2,"ctx":{"le":1}}]}
```

### `POST /predict` (deprecated)

Unchanged from 1.x. Request: `amount > 0`, `distance_from_home > 0`,
`use_chip ∈ {0, 1}`. Served by the v1 synthetic-data model, so `model_version`
is `"joblib_model"` or `"mock_rule"` — never `"joblib_model_v2"`.

```bash
curl -s -X POST https://mlops-fraud-detector.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"amount": 2500, "distance_from_home": 180.0, "use_chip": 0}'
# {"prediction":1,"is_fraud":true,"model_version":"joblib_model"}
```

### `GET /health`

```json
{
  "status": "healthy",
  "service": "Enterprise Real-Time Fraud Detection Engine",
  "version": "2.0.0",
  "model_version": "joblib_model_v2",
  "mock_mode": false,
  "endpoints": [
    {"method": "POST", "path": "/v2/predict", "status": "current",    "model_version": "joblib_model_v2"},
    {"method": "POST", "path": "/predict",    "status": "deprecated", "model_version": "joblib_model"},
    {"method": "GET",  "path": "/health",     "status": "current"},
    {"method": "GET",  "path": "/",           "status": "current"},
    {"method": "GET",  "path": "/docs",       "status": "current"}
  ]
}
```

---

## v1 (deprecated) vs v2 (current)

| | v1 — `POST /predict` | v2 — `POST /v2/predict` |
|---|---|---|
| Training data | `make_classification` synthetic (5,000 rows) | OpenML 45955 `card_transdata`, 1,000,000 real labelled rows |
| Features | `amount`, `distance_from_home`, `use_chip` | 7 behavioural features (table above) |
| Artifact | `models/fraud_model.joblib` — bare estimator | `models/fraud_model_v2.joblib` — `{"model", "features", "metrics", …}` dict |
| Fraud F1 on hold-out | 0.187 | 0.9999 |
| `model_version` when served | `joblib_model` / `mock_rule` | `joblib_model_v2` / `mock_rule` |
| OpenAPI | `deprecated: true`, description points to `/v2/predict` | current |
| Status | Frozen. Kept so existing integrations do not break. | Use this. |

The two schemas share no columns beyond `distance_from_home`, so a v2 model
cannot score a v1 payload. Rather than shim one into the other, each route is
served by its own artifact (or its own fallback rule) and reports its own
`model_version`. `GET /health` shows both at once. Nothing in v1's request or
response bytes changed in this release; the only visible difference is the
`deprecated` flag in `/openapi.json` and the strike-through in Swagger UI.

---

## Fallback Mock Model

`app/model_utils.py` wraps both classifiers in `FraudDetectionModel`. On
construction it tries `models/fraud_model_v2.joblib`, then
`models/fraud_model.joblib`:

| Artifacts loadable | `model_version` (top level / `/health`) | `/v2/predict` served by | `/predict` served by |
|---|---|---|---|
| v2 + v1 | `joblib_model_v2` | v2 forest | v1 forest |
| v2 only | `joblib_model_v2` | v2 forest | v1 rule → `mock_rule` |
| v1 only | `joblib_model` | v2 rule → `mock_rule` | v1 forest |
| neither | `mock_rule` (`mock_mode: true`) | v2 rule → `mock_rule` | v1 rule → `mock_rule` |

The v2 artifact is additionally rejected (with a logged warning) if it is not a
dict, has no `"model"`, or its `"features"` list differs from the API's expected
order — a mis-built artifact degrades to the rule rather than silently scoring
shuffled columns.

**Fallback rules**

```python
# v1
is_fraud = amount > 1000 and distance_from_home > 50.0 and use_chip == 0

# v2 — the strongest univariate fraud signals in card_transdata
is_fraud = (
    (ratio_to_median_purchase_price > 4.0 or distance_from_home > 100.0)
    and online_order == 1
    and used_pin_number == 0
)
```

**Why this matters**

- **CI stays green on day one.** A fresh clone or a runner that never trained a
  model can import the app, serve requests and pass all 34 tests. Tests that
  depend on a decision boundary are guarded on `v1_is_mock` / `v2_is_mock`, and
  the suite is run locally under all four artifact combinations above.
- **No silent degradation.** Every response carries `model_version` and
  `/health` carries `mock_mode` plus a per-route breakdown. The UI badge turns
  amber. The service never pretends to be something it is not.
- **Fail-safe, not fail-open.** Both rules still flag the highest-risk pattern
  rather than returning `0` for everything.

`model_version` is constrained to exactly three literal values —
`"joblib_model_v2"`, `"joblib_model"`, `"mock_rule"` — so it is safe to branch on.

---

## Model training & evaluation

Notebook: [`notebooks/train.ipynb`](notebooks/train.ipynb) (executed in place;
outputs are committed).

### Dataset

| | |
|---|---|
| Source | OpenML data id **45955**, `Credit_Card_Fraud_` (`card_transdata.csv`) |
| Fetch | `sklearn.datasets.fetch_openml(data_id=45955, as_frame=True, data_home="data")` — no Kaggle credentials |
| Licence | **Public Domain (CC0)** as declared on OpenML |
| Size | 1,000,000 rows × 7 features + `fraud` target; 8.74 % fraud |
| Cache | `data/` (git-ignored, ~20 MB compressed ARFF) |
| Download time | 18.5 s on first fetch, 1.6 s from cache. The notebook falls back to a stratified 200,000-row subsample only if the fetch exceeds 5 minutes; **that fallback was not needed — the model below is trained on the full 1,000,000 rows.** |

### Training

- Stratified 80/20 split (`random_state=42`): 800,000 train / 200,000 test.
- `RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=42)`
  — default 100 trees, no depth limit, no threshold tuning. Fit time 39 s.
- Artifact: `{"model", "features", "target", "metrics", "dataset", "subsampled",
  "n_rows_used", "sklearn_version", "trained_at", "fit_seconds"}` saved with
  `joblib.dump(..., compress=3)` → **1.14 MB**.

### Hold-out results (200,000 rows, 17,481 fraud)

Reported exactly as they came out of the notebook; nothing was tuned to improve them.

| Metric (fraud class) | Value |
|---|---|
| Precision | **1.0000** |
| Recall | **0.9997** |
| F1 | **0.9999** |
| PR-AUC (average precision) | **1.0000** |

Confusion matrix (rows = actual, columns = predicted):

| | pred legit | pred fraud |
|---|---|---|
| **actual legit** | 182,519 | 0 |
| **actual fraud** | 5 | 17,476 |

Feature importances (Gini):

| Feature | Importance |
|---|---|
| `ratio_to_median_purchase_price` | 0.542 |
| `distance_from_home` | 0.199 |
| `online_order` | 0.112 |
| `distance_from_last_transaction` | 0.081 |
| `used_pin_number` | 0.035 |
| `used_chip` | 0.025 |
| `repeat_retailer` | 0.007 |

Five missed fraud cases and zero false positives in 200,000 transactions is an
extraordinary result — and that is precisely why it should be read with the
[Limitations](#limitations) below, not as a claim about production fraud detection.

### v1 reference (synthetic, kept for `/predict`)

`RandomForestClassifier(n_estimators=100)` on 5,000 `make_classification` rows:
fraud precision 0.333, recall 0.130, F1 0.187. It proves the pipeline runs; it
does not model fraud.

Re-train either model:

```bash
pip install jupyter nbconvert
jupyter nbconvert --to notebook --execute notebooks/train.ipynb --inplace   # writes models/fraud_model_v2.joblib
```

---

## Limitations

**What these seven features cannot catch.** The model sees one transaction in
isolation, described by where it happened, how big it was relative to the
cardholder's median, and how it was authenticated. It has no view of:

- *Velocity and sequence* — ten small purchases in five minutes, or a card
  tested with a $1 charge before a large one. There is no time axis at all.
- *Merchant identity and category* — "repeat retailer" is a single bit; the
  model cannot know that a first purchase at a pharmacy and a first purchase at
  a crypto exchange carry different risk.
- *Account and device context* — new shipping address, new device fingerprint,
  recent password reset, login geography, account age.
- *Network effects* — a merchant or IP already linked to other confirmed fraud.
- *Amount in absolute terms* — a ratio of 6× is very different for a cardholder
  whose median is $8 versus $800.
- *Anything the features were engineered to summarise away* — `distance_from_home`
  presumes a reliable home address; `ratio_to_median_purchase_price` presumes
  enough history to have a median.

Fraud that looks like the cardholder — correct PIN, at home, usual amount,
account-takeover after credential theft — is invisible to this feature set by
construction.

**Why PR-AUC, not accuracy, on imbalanced data.** At 8.7 % prevalence a constant
"legitimate" predictor scores 91.3 % accuracy; at real-world prevalence closer to
0.1 % it scores 99.9 %. Accuracy is dominated by the easy negatives and says
nothing about the minority class you actually care about. ROC-AUC is better but
still rewards ranking a fraud case above the vast pool of trivial negatives.
Precision–recall curves and their area (average precision) are computed only on
how well positives are retrieved and how clean the retrieved set is — the two
quantities that translate directly into analyst workload and prevented loss.
That is why this README leads with fraud-class precision/recall/F1 and PR-AUC
and reports accuracy nowhere as a headline.

**This is a far gentler problem than production fraud.** Honest framing of the
near-perfect numbers above:

- *Prevalence.* 8.7 % fraud is roughly two orders of magnitude above real card
  fraud rates (~0.1 %). At 0.1 %, even a model with 99.9 % specificity would
  produce about as many false positives as true positives; the precision figure
  here would not survive the base-rate shift.
- *Clean, pre-engineered features.* The seven columns are exactly the kind of
  derived signals a mature fraud team spends years building. Here they arrive
  complete, correctly scaled, with no missing values, no leakage audit required
  and no upstream pipeline to break.
- *Labels are perfect and instant.* Production labels arrive weeks later via
  chargebacks, are incomplete (undetected fraud is labelled legitimate), and are
  themselves gamed.
- *No adversary.* The dataset is a static snapshot. Real fraudsters observe what
  gets declined and adapt within days; a model with zero false positives today
  is a model whose decision boundary is about to be mapped.
- *No drift, no seasonality, no policy changes.* Holiday spend, a new merchant
  category, a bank changing its 3-D Secure thresholds — none of that exists here.

The value of this repository is the *serving, validation and delivery
pipeline*: versioned artifacts with embedded feature contracts, an API that
cannot silently degrade, a UI that shows which model answered, and CI that
proves the container works before it ships. The model itself is a credible
demonstration on a public benchmark, not a fraud system.

---

## Testing & code quality

```bash
black --check .   # formatting
flake8 .          # linting (config in setup.cfg)
pytest -v         # 34 tests
```

`setup.cfg` pins `max-line-length = 88` and ignores `E203`/`W503` so black and
flake8 agree.

| Area | Tests | Asserts |
|---|---|---|
| Health & UI | `test_health_check`, `test_root_serves_html`, `test_static_assets_served` | `/health` JSON with per-route status; `/` is `text/html` with the page title and every v2 form control; CSS/JS served |
| v1 regression | `test_predict_valid_legit`, `test_predict_mock_fraud_trigger`, `test_predict_response_is_consistent`, `test_v1_regression_schema_unchanged` | `200`, exact key order, never served by `joblib_model_v2` |
| v1 validation | `…string_in_float_field`, `…negative_amount`, `…zero_amount`, `…bad_use_chip`, `…missing_field` ×3, `…empty_body` | `422` with the offending field in `loc` |
| v2 inference | `test_v2_predict_valid_legit`, `test_v2_predict_valid_fraud`, `test_v2_predict_response_is_consistent` | `200`, correct verdicts (guarded on `v2_is_mock`), deterministic |
| v2 validation | `…out_of_range_binary` ×4, `…string_in_float_field`, `…negative_distance`, `…missing_field` ×7, `test_v2_zero_values_are_valid` | `422` for each constraint; `ge=0` accepts exact zeros |
| OpenAPI | `test_openapi_schema_exposes_example`, `test_openapi_marks_v1_deprecated` | Pydantic v2 examples surface; `/predict` is `deprecated: true` and points to `/v2/predict`; `/` excluded from schema |

The suite is also run locally with each artifact hidden in turn (v2 only, v1
only, neither) — 34/34 in every configuration.

---

## CI/CD

Workflow: [`.github/workflows/ci-cd.yml`](.github/workflows/ci-cd.yml)

Triggers on every `push` and `pull_request` targeting `main`. One
`build-and-test` job on `ubuntu-latest`, Python 3.10, pip cache:

```
checkout ─► setup-python ─► pip install ─► black --check . ─► flake8 . ─► pytest -v
─► docker build
─► docker run … ls app/templates app/static models   (assets + both artifacts inside the image)
─► run container ─► curl /health   (assert joblib_model_v2, mock_mode:false)
                 ─► curl /         (assert Content-Type: text/html + page title + form)
                 ─► curl /static/* (assert 200)
                 ─► curl /v2/predict (assert prediction:1 for the fraud example)
                 ─► curl /predict    (assert v1 still answers with joblib_model)
```

The image-inspection step exists because the Dockerfile's `COPY app ./app` is
the only thing that ships the UI; if someone adds `app/static` to
`.dockerignore` the build still succeeds but the page 500s. CI catches that
before Render does. Concurrent runs on the same ref are cancelled.

Continuous deployment is handled by the platform: Render's `autoDeploy: true`
redeploys on every push to `main`.

---

## Deployment

### Render (primary) — Blueprint

`render.yaml` declares one Docker web service (`plan: free`, `region: frankfurt`,
`healthCheckPath: /health`, `autoDeploy: true`).

1. <https://dashboard.render.com> → **New → Blueprint** → select this repo → **Apply**
   (or click the *Deploy to Render* button at the top of this file).
2. Render builds the Dockerfile and injects `$PORT`; the shell-form `CMD` picks it up.
3. Every push to `main` redeploys automatically.

For redeploys from the terminal, copy the service's *Deploy Hook* URL
(Settings → Deploy Hook):

```bash
export RENDER_DEPLOY_HOOK_URL="https://api.render.com/deploy/srv-xxxxxxxxxxxx?key=yyyyyyyy"
curl -X POST "$RENDER_DEPLOY_HOOK_URL"
```

### Hugging Face Spaces — always-on demo

The YAML frontmatter at the top of this README makes the repository a valid
Docker Space (`sdk: docker`, `app_port: 8000`). Free CPU Spaces do not sleep
the way Render's free tier does.

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli repo create mlops-fraud-detector --type space --space_sdk docker
git remote add hf https://huggingface.co/spaces/<your-hf-username>/mlops-fraud-detector
git push hf main
```

### Railway (alternative)

```bash
railway login && railway init && railway up && railway domain
```

---

## Project layout

```
mlops-fraud-detector/
├── .github/workflows/ci-cd.yml   # lint → test → docker build → image inspection → smoke test
├── app/
│   ├── __init__.py
│   ├── main.py                   # FastAPI app: /, /health, /v2/predict, /predict (deprecated)
│   ├── model_utils.py            # FraudDetectionModel: v2 → v1 → mock cascade, singleton
│   ├── static/
│   │   ├── style.css             # UI stylesheet (light theme, system fonts)
│   │   └── app.js                # health badge, validation, fetch, verdict rendering
│   └── templates/
│       └── index.html            # Jinja2 page served at /
├── data/                         # fetch_openml cache (git-ignored)
├── models/
│   ├── .gitkeep
│   ├── fraud_model.joblib        # v1 (synthetic) — kept for POST /predict
│   └── fraud_model_v2.joblib     # v2 dict {"model", "features", "metrics", ...}
├── notebooks/train.ipynb         # v2 training on OpenML 45955, executed in place
├── tests/
│   ├── __init__.py
│   └── test_api.py               # 34 contract tests, mock-aware
├── Dockerfile                    # python:3.10-slim, non-root, cached deps layer
├── .dockerignore  .gitignore  .gitattributes
├── setup.cfg                     # flake8 config aligned with black
├── render.yaml                   # Render Blueprint (health check on /health)
├── requirements.txt              # exact pins, resolved for Python 3.10
└── README.md                     # this file (doubles as the HF Space card)
```

---

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `PORT` | `8000` | Port uvicorn binds to. Injected by Render / Railway / HF Spaces. |
| `LOG_LEVEL` | `INFO` | Python logging level for the app and model loader. |

---

## License

Code: MIT. Data: OpenML 45955 is declared Public Domain (CC0) by its uploader;
no cardholder-identifying data is present in the dataset or stored by this
service.

# Day 12: Model Deployment Mini-Project (FastAPI)

Every project so far in this series lived inside a notebook. A notebook only works while a person is sitting there clicking "run." This project takes the Day 8 to Day 11 loan default model and gives it a doorbell, so any program on the same machine, or on a network, can knock on the door and get a prediction back, with no notebook involved.

## Explained like you are five

Think of the trained model as a very smart puppet who is very good at guessing one thing: will this person pay back their loan. But the puppet only knows how to talk to the person who built it, inside a private room (the notebook).

This project builds a **doorbell** for the puppet's room. Now anyone can walk up, ring the doorbell with a description of a loan applicant, wait one second, and get an answer shouted back through the door: "high risk" or "low risk," and a number for how sure the puppet is.

The doorbell also has rules taped to it. If you ring it with a silly description, like a credit score of 999 (credit scores only go up to 850), the doorbell refuses to even ask the puppet, and tells you what you did wrong instead. That way the puppet never has to look at garbage.

## Problem statement

Take a trained loan default model and expose it as a small, testable HTTP API with a `/predict` endpoint, input validation, a health check, and automated tests, runnable entirely on a local machine with no cloud hosting required.

## Dataset / model

Reuses the same made-up loan applicant data and LightGBM setup from Day 11 (`scripts/train_model.py` regenerates it and retrains from scratch, so the whole pipeline is reproducible from nothing but this repo). Eight features go in: age, annual income, employment years, credit score, loan amount, debt-to-income, number of late payments, and whether there is a cosigner.

## Approach

1. **`scripts/train_model.py`** builds the synthetic dataset, trains a LightGBM classifier, and saves it plus its feature order and metrics into one `joblib` bundle at `model/loan_default_model.joblib`.
2. **`app/model_service.py`** loads that bundle once and exposes a small `predict_one` / `predict_many` interface, kept separate from any web framework code so it can be tested on its own.
3. **`app/schemas.py`** defines exactly what a valid request and response look like with Pydantic, so FastAPI can reject bad input automatically (wrong type, out-of-range credit score, negative income) before it ever reaches the model.
4. **`app/main.py`** wires it into FastAPI: `/predict` for one applicant, `/predict/batch` for several at once, `/health` for a status check, all documented automatically at `/docs`.
5. **`tests/test_api.py`** is an automated pytest suite that checks the whole thing end to end, in memory, with no server needed to run it.
6. **`scripts/smoke_test.py`** is a manual script for hitting a real, running server with real HTTP requests, for a live sanity check.

## Results

Model performance (same family and same numbers as Day 11's baseline model, reported again here since this is the deployed artifact):

| Metric | Value |
|---|---|
| ROC AUC | 0.783 |
| PR AUC | 0.553 |
| F1 at 0.5 threshold | 0.477 |
| Train / test rows | 6,000 / 2,000 |

API test suite: **9 of 9 tests pass** (`pytest -v`), covering a healthy response shape, sensible relative ordering between a safe and a risky applicant, rejected malformed input (missing field, out-of-range credit score, negative income), and batch predictions matching individual calls exactly.

## Try it yourself

Start the server:

```bash
pip install -r requirements.txt
python scripts/train_model.py        # only needed once, to create model/loan_default_model.joblib
uvicorn app.main:app --reload
```

Then, in another terminal:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 34, "annual_income": 62000, "employment_years": 6.5, "credit_score": 610, "loan_amount": 42000, "debt_to_income": 0.38, "num_late_payments": 3, "has_cosigner": 0}'
```

Real response from this exact request:

```json
{
  "default_probability": 0.7573,
  "will_default": true,
  "risk_band": "high",
  "decision_threshold": 0.5,
  "model_version": "1.0.0"
}
```

A safe applicant next to a risky one, sent together as a batch:

```bash
curl -X POST http://127.0.0.1:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[
    {"age": 45, "annual_income": 120000, "employment_years": 15, "credit_score": 790, "loan_amount": 15000, "debt_to_income": 0.12, "num_late_payments": 0, "has_cosigner": 1},
    {"age": 24, "annual_income": 28000, "employment_years": 0.5, "credit_score": 520, "loan_amount": 24000, "debt_to_income": 0.65, "num_late_payments": 6, "has_cosigner": 0}
  ]'
```

Real response:

```json
{
  "predictions": [
    {"default_probability": 0.0025, "will_default": false, "risk_band": "low", "decision_threshold": 0.5, "model_version": "1.0.0"},
    {"default_probability": 0.9871, "will_default": true, "risk_band": "high", "decision_threshold": 0.5, "model_version": "1.0.0"}
  ]
}
```

Sending a credit score of 999 (invalid, since the range is 300 to 850) gets rejected before it reaches the model:

```json
{
  "detail": [
    {"type": "less_than_equal", "loc": ["body", "credit_score"], "msg": "Input should be less than or equal to 850", "ctx": {"le": 850.0}}
  ]
}
```

That response comes back with HTTP status `422`, not `200`, so a caller can tell programmatically that its input was bad rather than trusting a possibly wrong prediction.

You can also open `http://127.0.0.1:8000/docs` in a browser for an interactive page that lets you send requests by clicking buttons, no curl needed.

## Frontend (frontend/index.html)

For a friendlier way to try the model than curl or the auto-generated docs page, `frontend/index.html` is a small standalone page: a form on the left for the applicant's details, and a live assessment on the right, with a probability figure, a colored risk gauge, and a plain-English verdict sentence.

It is a plain HTML file with no build step and no framework, so it just needs to be opened in a browser while the API is running:

```bash
uvicorn app.main:app --reload   # keep this running in one terminal
```

Then open `frontend/index.html` directly in a browser (double-click it, or `open frontend/index.html` on macOS). It talks to `http://127.0.0.1:8000`, the same address the curl examples above use. The API has CORS enabled specifically so this page, opened as a local file, is allowed to call it.

The page includes three one-click example applicants (safe, risky, borderline) so you can see contrasting predictions immediately, validates every field itself before sending anything (with the same ranges as the API), and shows a clear message if the API isn't running rather than failing silently.

## Why this matters (deployment vs. modeling)

A notebook proves you can train a model. An API proves you can hand that model to someone else, safely:

- **Validation at the door.** Bad input gets a clear `422` error instead of a silent wrong answer or a crash.
- **Loaded once, reused many times.** The model loads from disk one time at startup, not on every request, which is the difference between a fast API and a slow one.
- **Separation of concerns.** `model_service.py` knows nothing about HTTP, and `main.py` knows nothing about LightGBM internals. Either one can be tested, swapped, or reused on its own.
- **Automated tests, not manual clicking.** `pytest` re-checks all of this in under two seconds, every time the code changes.
- **No cloud needed to prove it works.** Everything above runs on a laptop. The same app, unchanged, is what would go into a Docker container for a real deployment later.
- **A frontend that talks to your own API, not someone else's.** `frontend/index.html` is a good example of one system (a browser page) calling another (your FastAPI service) over a real network boundary, which is closer to how deployed systems actually get used than calling functions directly in a notebook.

## Honest limits

- The model itself is the same synthetic-data model from Day 11; deploying it does not make its predictions more trustworthy, only more reachable.
- No authentication, no rate limiting, no HTTPS. Fine for a local demo, not fine for production.
- Single `joblib` file with no model registry or versioned rollback; `model_version` is a hardcoded string, not tied to an actual versioning system.
- No logging of requests or predictions, so nothing here would catch model drift over time.

## What I would improve next

1. Add authentication (an API key header) before this ever touches real data.
2. Log every request and prediction to a file or database, so drift can be measured later (ties into Day 24's evaluation-harness thinking).
3. Containerize with Docker so "works on my machine" becomes "works anywhere."
4. Add the SHAP explanation from Day 11 as an optional field in the response, so a caller can get a reason along with the number.
5. Swap the hardcoded threshold and version string for values read from the model bundle's own metadata at load time.

## Repository layout

```
day-12-model-deployment-fastapi/
  README.md
  requirements.txt
  .gitignore
  app/
    main.py            FastAPI app: routes, startup, error handling, CORS
    model_service.py    Model loading + prediction logic, framework-free
    schemas.py           Pydantic request/response models and validation
  frontend/
    index.html            Standalone interactive frontend (no build step)
  model/
    loan_default_model.joblib   Trained model bundle (model + feature order + metrics)
  scripts/
    train_model.py       Regenerates data, trains, and saves the model
    smoke_test.py         Manual check against a live running server
  tests/
    test_api.py            Automated pytest suite (9 tests, in-memory)
```

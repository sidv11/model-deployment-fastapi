"""
Day 12: wrap the loan default model in a tiny FastAPI app.

Explained simply: Day 8 through Day 11 built a robot that lives
inside a notebook. A notebook is like a robot that only works while
you are standing right next to it. This file gives the robot a
doorbell. Now anyone (or any other program) can knock on the door
with one applicant's details, and the robot answers with a
prediction, without ever seeing the notebook or the training code.

Run it:
    uvicorn app.main:app --reload

Then visit http://127.0.0.1:8000/docs for a page where you can try
it by clicking buttons, or see the README for curl examples.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.model_service import ModelNotLoadedError, model_service
from app.schemas import (
    BatchPredictionResponse,
    HealthResponse,
    LoanApplicant,
    PredictionResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once when the server starts, not on every request.
    # Loading from disk on every request would make each call slower
    # and is unnecessary work since the model does not change between calls.
    model_service.load()
    yield


app = FastAPI(
    title="Loan Default Predictor",
    description="A minimal API wrapping a LightGBM loan default model. Day 12 of a 30-day GitHub build.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the plain-HTML frontend (frontend/index.html, opened directly as a
# file:// page or served from any local port) to call this API. Wide open on
# purpose since this only ever runs on a local machine for a demo, never in
# production — a real deployment would lock this down to a specific origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["meta"])
def root():
    return {"message": "Loan default predictor is running. See /docs for the interactive API page."}


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    """A quick 'are you alive and ready' check, useful for monitoring or a load balancer."""
    return HealthResponse(
        status="ok" if model_service.is_loaded else "model not loaded",
        model_version=model_service.model_version,
        model_loaded=model_service.is_loaded,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(applicant: LoanApplicant):
    """Score one loan applicant and return a default probability and risk band."""
    try:
        result = model_service.predict_one(applicant.model_dump())
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return PredictionResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["prediction"])
def predict_batch(applicants: list[LoanApplicant]):
    """Score a list of applicants in one call, instead of one HTTP round trip each."""
    if not applicants:
        raise HTTPException(status_code=400, detail="Send at least one applicant.")
    try:
        results = model_service.predict_many([a.model_dump() for a in applicants])
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return BatchPredictionResponse(predictions=[PredictionResponse(**r) for r in results])

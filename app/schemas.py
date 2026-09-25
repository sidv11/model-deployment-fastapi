"""
The shape of what goes in and what comes out of the API.

Think of this file as a bouncer at a door. Nothing gets into the model
unless it matches this shape, and nothing goes back to the caller
unless it matches this shape either.
"""

from pydantic import BaseModel, Field, ConfigDict


class LoanApplicant(BaseModel):
    """One loan applicant, described with the same eight clues the model was trained on."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 34,
                "annual_income": 62000,
                "employment_years": 6.5,
                "credit_score": 610,
                "loan_amount": 42000,
                "debt_to_income": 0.38,
                "num_late_payments": 3,
                "has_cosigner": 0,
            }
        }
    )

    age: float = Field(..., ge=18, le=100, description="Applicant age in years")
    annual_income: float = Field(..., gt=0, description="Annual income in the same currency as loan_amount")
    employment_years: float = Field(..., ge=0, le=60, description="Years at current job")
    credit_score: float = Field(..., ge=300, le=850, description="Credit score, standard 300-850 range")
    loan_amount: float = Field(..., gt=0, description="Amount being requested")
    debt_to_income: float = Field(..., ge=0, le=2, description="Existing debt divided by annual income")
    num_late_payments: int = Field(..., ge=0, le=50, description="Late payments in the last 12 months")
    has_cosigner: int = Field(..., ge=0, le=1, description="1 if a cosigner is on the loan, else 0")


class PredictionResponse(BaseModel):
    default_probability: float = Field(..., description="Model's estimated probability of default, 0 to 1")
    will_default: bool = Field(..., description="default_probability >= decision_threshold")
    risk_band: str = Field(..., description="low, medium, or high, for a quick human read")
    decision_threshold: float
    model_version: str


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_version: str
    model_loaded: bool

"""
A manual check against a server that is actually running, as opposed
to tests/test_api.py which talks to the app in memory.

Start the server first in another terminal:
    uvicorn app.main:app --reload

Then run this script:
    python scripts/smoke_test.py

It prints what it sends and what it gets back, so you can see the
whole round trip with real numbers.
"""

import sys

import requests

BASE_URL = "http://127.0.0.1:8000"

APPLICANT = {
    "age": 29,
    "annual_income": 48000,
    "employment_years": 3,
    "credit_score": 615,
    "loan_amount": 30000,
    "debt_to_income": 0.42,
    "num_late_payments": 4,
    "has_cosigner": 0,
}


def main() -> None:
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=5)
    except requests.exceptions.ConnectionError:
        print(f"Could not reach {BASE_URL}. Is the server running? (uvicorn app.main:app --reload)")
        sys.exit(1)

    print("GET /health ->", health.status_code, health.json())

    prediction = requests.post(f"{BASE_URL}/predict", json=APPLICANT, timeout=5)
    print("\nPOST /predict")
    print("  sent:", APPLICANT)
    print("  status:", prediction.status_code)
    print("  received:", prediction.json())

    bad_applicant = dict(APPLICANT)
    bad_applicant["credit_score"] = 999
    rejected = requests.post(f"{BASE_URL}/predict", json=bad_applicant, timeout=5)
    print("\nPOST /predict with an invalid credit_score (999)")
    print("  status:", rejected.status_code, "(expected 422, meaning it was correctly rejected)")


if __name__ == "__main__":
    main()

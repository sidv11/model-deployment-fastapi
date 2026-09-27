"""
Day 12 frontend, Streamlit version.

A plain Streamlit app that calls the same FastAPI service used by the
curl examples in the README. Streamlit is just another client of the
API here, exactly like curl or a browser hitting /docs, it does not
load or touch the model directly.

Run (with the API already running in another terminal):
    uvicorn app.main:app --reload
    streamlit run frontend_streamlit/app.py
"""

import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Loan Default Predictor", layout="centered")

FIELDS = [
    ("age", "Age", 18, 100, 1, "%d"),
    ("annual_income", "Annual income", 1, 10_000_000, 1000, "%d"),
    ("employment_years", "Years at current job", 0.0, 60.0, 0.5, "%.1f"),
    ("credit_score", "Credit score", 300, 850, 1, "%d"),
    ("loan_amount", "Loan amount requested", 1, 10_000_000, 1000, "%d"),
    ("debt_to_income", "Debt-to-income ratio", 0.0, 2.0, 0.01, "%.2f"),
    ("num_late_payments", "Late payments, last 12 months", 0, 50, 1, "%d"),
]

PRESETS = {
    "Safe example": dict(age=45, annual_income=120000, employment_years=15.0, credit_score=790,
                          loan_amount=15000, debt_to_income=0.12, num_late_payments=0, has_cosigner=True),
    "Risky example": dict(age=24, annual_income=28000, employment_years=0.5, credit_score=520,
                           loan_amount=24000, debt_to_income=0.65, num_late_payments=6, has_cosigner=False),
    "Borderline example": dict(age=29, annual_income=48000, employment_years=3.0, credit_score=615,
                                loan_amount=30000, debt_to_income=0.42, num_late_payments=4, has_cosigner=False),
}

DEFAULTS = dict(
    age=34, annual_income=62000, employment_years=6.5, credit_score=610,
    loan_amount=42000, debt_to_income=0.38, num_late_payments=3, has_cosigner=False,
)

for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)
st.session_state.setdefault("result", None)
st.session_state.setdefault("error", None)


def apply_preset(values: dict):
    for k, v in values.items():
        st.session_state[k] = v
    st.session_state["result"] = None
    st.session_state["error"] = None


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("Loan Default Predictor")
st.caption(
    "Day 12 of a 30-day GitHub build. This page calls the FastAPI service "
    f"at `{API_BASE}` — start it first with `uvicorn app.main:app --reload`."
)

st.subheader("Applicant details")

preset_cols = st.columns(3)
for col, label in zip(preset_cols, PRESETS):
    col.button(label, use_container_width=True, on_click=apply_preset, args=(PRESETS[label],))

with st.form("applicant_form"):
    left, right = st.columns(2)
    for i, (key, label, lo, hi, step, fmt) in enumerate(FIELDS):
        target = left if i % 2 == 0 else right
        target.number_input(label, min_value=lo, max_value=hi, step=step, format=fmt, key=key)

    st.checkbox("Has a cosigner", key="has_cosigner")
    submitted = st.form_submit_button("Assess this applicant", type="primary", use_container_width=True)

if submitted:
    payload = {k: st.session_state[k] for k, *_ in FIELDS}
    payload["has_cosigner"] = 1 if st.session_state["has_cosigner"] else 0
    with st.spinner("Asking the model..."):
        try:
            resp = requests.post(f"{API_BASE}/predict", json=payload, timeout=5)
            if resp.ok:
                st.session_state["result"] = resp.json()
                st.session_state["error"] = None
            else:
                body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                detail = body.get("detail")
                if isinstance(detail, list):
                    detail = " ".join(d.get("msg", "") for d in detail)
                st.session_state["error"] = f"The API rejected this request (HTTP {resp.status_code}). {detail or ''}"
                st.session_state["result"] = None
        except requests.exceptions.RequestException:
            st.session_state["error"] = (
                f"Couldn't reach {API_BASE}. Is the server running? "
                "Start it with `uvicorn app.main:app --reload` in the project folder, then try again."
            )
            st.session_state["result"] = None

# ----------------------------------------------------------------------
# Result
# ----------------------------------------------------------------------
st.divider()
st.subheader("Assessment")

result = st.session_state["result"]
error = st.session_state["error"]

if error:
    st.error(error)

elif result:
    band = result["risk_band"]
    pct = result["default_probability"] * 100
    threshold_pct = result["decision_threshold"] * 100

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Default probability", f"{pct:.1f}%")
    with col2:
        if band == "low":
            st.success(f"**LOW RISK** — below the {threshold_pct:.0f}% line the model uses to flag concern.")
        elif band == "medium":
            st.warning(f"**MEDIUM RISK** — under the {threshold_pct:.0f}% line, but close enough to warrant a second look.")
        else:
            st.error(f"**HIGH RISK** — at or above the {threshold_pct:.0f}% line, flagged for default.")

    st.progress(min(int(pct), 100))

    with st.expander("Raw API response"):
        st.json(result)

else:
    st.info("Fill in the form above and submit it to see the model's prediction.")

st.caption("Serves the model trained in `scripts/train_model.py`. Every field is validated twice: once here, once again by the API itself.")

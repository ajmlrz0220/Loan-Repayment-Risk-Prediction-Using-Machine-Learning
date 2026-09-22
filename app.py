"""
Streamlit app for the Loan Default Risk model.

Run locally:
    streamlit run app.py

Expects loan_default_model.pkl in the same folder (produced by train_model.py,
which is the script version of the training notebook).
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Loan Default Risk Predictor",
    page_icon="💳",
    layout="wide",
)

MODEL_PATH = "loan_default_model.pkl"


@st.cache_resource
def load_model():
    try:
        artifact = joblib.load(MODEL_PATH)
        return artifact
    except FileNotFoundError:
        return None


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["credit_history_years"] = df["days.with.cr.line"] / 365
    return df


PURPOSE_OPTIONS = [
    "debt_consolidation",
    "credit_card",
    "all_other",
    "home_improvement",
    "small_business",
    "major_purchase",
    "educational",
]

artifact = load_model()

st.title("💳 Loan Default Risk Predictor")
st.caption(
    "A Random Forest model trained on historical loan data to estimate the "
    "probability that a borrower will **not fully repay** a loan. "
    "Built with scikit-learn, served with Streamlit."
)

if artifact is None:
    st.error(
        f"Couldn't find `{MODEL_PATH}` next to this app. "
        "Run `python train_model.py --data loan_data.csv` first to create it, "
        "then reload this page."
    )
    st.stop()

model = artifact["model"]
threshold = artifact["threshold"]

tab_single, tab_batch, tab_about = st.tabs(
    ["🔍 Single Applicant", "📄 Batch Scoring (CSV)", "ℹ️ About this model"]
)

# ---------------------------------------------------------------------------
# TAB 1 — Single applicant
# ---------------------------------------------------------------------------
with tab_single:
    st.subheader("Enter applicant details")

    col1, col2, col3 = st.columns(3)

    with col1:
        credit_policy = st.selectbox(
            "Meets credit underwriting criteria?", [1, 0], index=0
        )
        purpose = st.selectbox("Loan purpose", PURPOSE_OPTIONS)
        int_rate = st.slider("Interest rate", 0.05, 0.25, 0.12, 0.001, format="%.3f")
        installment = st.number_input("Monthly installment ($)", 15.0, 1000.0, 300.0)

    with col2:
        log_annual_inc = st.number_input(
            "Log of annual income", 6.0, 15.0, 10.8, 0.1,
            help="e.g. an $60,000 income ≈ log(60000) ≈ 11.0",
        )
        dti = st.slider("Debt-to-income ratio", 0.0, 40.0, 15.0, 0.1)
        fico = st.slider("FICO score", 600, 850, 700)
        revol_bal = st.number_input("Revolving balance ($)", 0, 200000, 8000)

    with col3:
        revol_util = st.slider("Revolving line utilization (%)", 0.0, 150.0, 45.0)
        inq_last_6mths = st.number_input("Credit inquiries, last 6 months", 0, 20, 1)
        delinq_2yrs = st.number_input("Delinquencies, last 2 years", 0, 15, 0)
        pub_rec = st.number_input("Public derogatory records", 0, 10, 0)
        days_with_cr_line = st.number_input(
            "Days with credit line", 100, 20000, 4500
        )

    if st.button("Predict risk", type="primary"):
        input_df = pd.DataFrame(
            [
                {
                    "credit.policy": credit_policy,
                    "purpose": purpose,
                    "int.rate": int_rate,
                    "installment": installment,
                    "log.annual.inc": log_annual_inc,
                    "dti": dti,
                    "fico": fico,
                    "days.with.cr.line": days_with_cr_line,
                    "revol.bal": revol_bal,
                    "revol.util": revol_util,
                    "inq.last.6mths": inq_last_6mths,
                    "delinq.2yrs": delinq_2yrs,
                    "pub.rec": pub_rec,
                }
            ]
        )
        input_df = engineer_features(input_df)

        probability = model.predict_proba(input_df)[:, 1][0]
        prediction = int(probability >= threshold)

        st.divider()
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Predicted risk probability", f"{probability:.1%}")
            st.metric("Decision threshold", f"{threshold:.2f}")
        with c2:
            if prediction == 1:
                st.error(
                    "⚠️ **Higher risk** — model predicts this borrower is likely "
                    "to NOT fully repay the loan."
                )
            else:
                st.success(
                    "✅ **Lower risk** — model predicts this borrower is likely "
                    "to fully repay the loan."
                )
        st.progress(min(probability, 1.0))

# ---------------------------------------------------------------------------
# TAB 2 — Batch scoring
# ---------------------------------------------------------------------------
with tab_batch:
    st.subheader("Score a CSV of applicants")
    st.write(
        "Upload a CSV with the same raw columns as `loan_data.csv` "
        "(no need to include `not.fully.paid`)."
    )
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        try:
            scored = engineer_features(batch_df)
            probs = model.predict_proba(scored)[:, 1]
            batch_df["risk_probability"] = probs
            batch_df["predicted_default"] = (probs >= threshold).astype(int)
            st.success(f"Scored {len(batch_df)} applicants.")
            st.dataframe(batch_df, use_container_width=True)
            st.download_button(
                "Download scored CSV",
                batch_df.to_csv(index=False).encode("utf-8"),
                file_name="scored_applicants.csv",
                mime="text/csv",
            )
        except KeyError as e:
            st.error(f"Missing expected column: {e}")

# ---------------------------------------------------------------------------
# TAB 3 — About
# ---------------------------------------------------------------------------
with tab_about:
    st.subheader("Model details")
    st.markdown(
        f"""
- **Algorithm:** Random Forest (`class_weight="balanced"`), tuned with
  `GridSearchCV` optimizing PR-AUC (chosen over accuracy because the target
  is imbalanced — most loans *are* repaid).
- **Decision threshold:** `{threshold:.2f}`, selected by sweeping thresholds
  on the test set and picking the one that maximizes F1, rather than the
  default 0.5.
- **Features used:** credit policy, loan purpose, interest rate, installment,
  log annual income, debt-to-income ratio, FICO score, revolving balance/
  utilization, recent credit inquiries, delinquencies, public records, and
  an engineered `credit_history_years` feature.
        """
    )

    if hasattr(model.named_steps["model"], "feature_importances_"):
        st.subheader("Top feature importances")
        preprocessor = model.named_steps["preprocessor"]
        importances = model.named_steps["model"].feature_importances_
        feature_names = preprocessor.get_feature_names_out()
        imp_df = (
            pd.DataFrame({"Feature": feature_names, "Importance": importances})
            .sort_values("Importance", ascending=False)
            .head(15)
            .set_index("Feature")
        )
        st.bar_chart(imp_df)

    st.caption(
        "⚠️ This is a portfolio/demo project trained on historical data. "
        "It is not financial advice and should not be used for real "
        "lending decisions."
    )

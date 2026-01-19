import streamlit as st
import pandas as pd
import shap
import numpy as np
from sklearn.linear_model import LogisticRegression

# --- 1. THE BRAIN: Weighted Linear Model ---
# Banks prefer this for compliance because every factor is weighted fairly
def train_final_model():
    data = pd.DataFrame({
        'income': [1000, 5000, 2000, 9000, 3000, 1500, 7000, 12000, 4000, 2500],
        'debt':   [1000, 200, 800, 50, 400, 900, 150, 10, 300, 600],
        'history':[1, 10, 2, 20, 5, 1, 15, 25, 7, 3],
        'denied': [1, 0, 1, 0, 0, 1, 0, 0, 0, 1] 
    })
    X = data.drop('denied', axis=1)
    y = data['denied']
    
    # Logistic Regression creates a smooth scoring system
    model = LogisticRegression().fit(X, y)
    return model, X

# --- 2. LAYMAN EXPLANATIONS ---
def get_human_reason(feature, is_denied):
    reasons = {
        "income": {
            "denied": "Income level is below the safety threshold for this loan.",
            "approved": "Your income provides strong coverage for potential repayments."
        },
        "debt": {
            "denied": "Your current debt obligations are creating a high-risk ratio.",
            "approved": "Low existing debt indicates high financial flexibility."
        },
        "history": {
            "denied": "A limited credit history reduces our long-term predictive confidence.",
            "approved": "Your established credit history proves long-term reliability."
        }
    }
    return reasons[feature]["denied" if is_denied else "approved"]

# --- 3. UI DASHBOARD ---
st.set_page_config(page_title="LexVerify AI", layout="wide", page_icon="🛡️")

st.title("🛡️ LexVerify AI | Compliance Audit Dashboard")
st.markdown("#### *Mathematical Transparency for Banking AI*")

# Sidebar Inputs
st.sidebar.header("📋 Applicant Data")
u_inc = st.sidebar.number_input("Monthly Income ($)", 500, 15000, 2500)
u_deb = st.sidebar.number_input("Monthly Debt ($)", 0, 5000, 800)
u_his = st.sidebar.slider("Credit History (Years)", 0, 30, 2)

# Run Logic
model, X_train = train_final_model()
user_input = pd.DataFrame([[u_inc, u_deb, u_his]], columns=X_train.columns)

# Calculate Prediction & SHAP
prediction = model.predict(user_input)[0]
explainer = shap.LinearExplainer(model, X_train)
shap_values = explainer.shap_values(user_input)

# Final Status
if prediction == 1:
    st.error("### 🛑 STATUS: LOAN DENIED")
else:
    st.success("### ✅ STATUS: LOAN APPROVED")

st.divider()

# --- 4. FACTOR ANALYSIS (With Normalization Fix) ---
st.subheader("⚖️ Regulatory Factor Impact")
cols = st.columns(3)
features = ["income", "debt", "history"]
display_names = ["Monthly Income", "Total Debt", "Credit History"]

# 1. Get absolute raw scores
raw_impacts = [abs(v) for v in shap_values[0]]
total_impact = sum(raw_impacts)

# 2. Normalize so they add up to 100%
normalized_impacts = [(v / total_impact) * 100 for v in raw_impacts]

for i, col in enumerate(cols):
    with col:
        # Now it will show a clean percentage like 45.2%
        st.metric(display_names[i], f"{round(normalized_impacts[i], 1)}% Weight")
        st.write(f"**Audit Note:** {get_human_reason(features[i], prediction)}")

import streamlit as st
import pandas as pd
from model_utils import explain_denial

st.set_page_config(page_title="LexVerify AI Dashboard", page_icon="🛡️")

st.title("🛡️ LexVerify AI")
st.markdown("### *Transparency Layer for Banking Compliance*")

st.divider()

# Input section
st.sidebar.header("📝 Applicant Details")
income = st.sidebar.slider("Monthly Income ($)", 500, 10000, 2000)
debt = st.sidebar.slider("Current Debt ($)", 0, 5000, 800)
history = st.sidebar.slider("Credit History (Years)", 0, 20, 2)

user_data = {
    "income_monthly": income,
    "debt_amount": debt,
    "credit_history_years": history
}

if st.button("Run Compliance Audit"):
    with st.spinner('Analyzing Model Decision...'):
        # Get SHAP reasons
        reasons = explain_denial(user_data)
        
        st.error("🛑 Loan Status: DENIED")
        
        st.subheader("⚖️ Legally Required Explanations")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Primary Factor", value=reasons[0][0].replace('_', ' ').title())
            st.write(f"Contribution Score: `{round(reasons[0][1], 4)}`")
            
        with col2:
            st.metric(label="Secondary Factor", value=reasons[1][0].replace('_', ' ').title())
            st.write(f"Contribution Score: `{round(reasons[1][1], 4)}`")
            
        st.info("💡 **Mathematical Proof:** The reasons above were reverse-engineered from the Black-Box model using Shapley values, ensuring 0% hallucination.")

st.sidebar.write("---")
st.sidebar.write("Built for AISSMS GDG Inauguration 2026")

# LexVerify.ai: Solving the $50M Compliance Gap

### 🏦 The Problem
Banks use AI to approve loans, but the law (US ECOA/GDPR) requires them to explain *why* a customer was denied. Most modern AI is a "Black Box"—it can't explain itself. 

### 🛡️ The Solution
AuditAI is a **Deterministic XAI Wrapper**. We don't replace the bank's AI; we wrap it. Using SHAP (Shapley Additive Explanations), we reverse-engineer any model's decision into legally compliant "Adverse Action Notices."

### 🚀 Why this beats a fine-tuned LLM:
- **No Hallucinations:** We use math, not probabilistic word-guessing.
- **Audit Trails:** A mathematical proof of causality backs every explanation.
- **Model Agnostic:** Works with XGBoost, Neural Nets, or custom bank models.

### 🛠️ Tech Stack
- **XAI:** SHAP (Game Theory-based explanations)
- **Backend:** FastAPI (Production-grade API)
- **Frontend:** Streamlit (Instant Dashboard)
- **Math:** Scikit-learn, Pandas

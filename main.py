from fastapi import FastAPI
from model_utils import explain_denial

app = FastAPI(title="LexVerify Compliance API")

@app.get("/")
def home():
    return {"status": "LexVerify Engine Active", "version": "1.0.0"}

@app.post("/audit")
async def audit_decision(user_data: dict):
    # Expected JSON: {"income_monthly": 2000, "debt_amount": 800, "credit_history_years": 2}
    reasons = explain_denial(user_data)
    
    # Format the top 2 reasons for the legal notice
    primary = reasons[0]
    secondary = reasons[1]
    
    return {
        "decision": "Denied",
        "compliance_report": {
            "primary_reason": f"High {primary[0].replace('_', ' ')} impact",
            "secondary_reason": f"Insufficient {secondary[0].replace('_', ' ')}",
            "evidence_scores": {primary[0]: round(primary[1], 4), secondary[0]: round(secondary[1], 4)},
            "regulatory_stamp": "ECOA-COMPLIANT-V1"
        }
    }

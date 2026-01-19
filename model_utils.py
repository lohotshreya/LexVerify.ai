import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier

# 1. Create a "Fake" Bank Model for the demo
def get_mock_model_and_data():
    data = pd.DataFrame({
        'income_monthly': [5000, 2000, 8000, 1500, 6000, 2200, 4500, 1200],
        'debt_amount': [200, 800, 100, 900, 300, 700, 400, 950],
        'credit_history_years': [10, 2, 15, 1, 8, 3, 7, 1],
        'denied': [0, 1, 0, 1, 0, 1, 0, 1] # 1 = Denied, 0 = Approved
    })
    X = data.drop('denied', axis=1)
    y = data['denied']
    
    # Simple Black-Box model
    model = RandomForestClassifier(n_estimators=10).fit(X, y)
    return model, X

# 2. The XAI Engine: Uses SHAP to find the 'Why'
def explain_denial(user_input_dict):
    model, X = get_mock_model_and_data()
    
    # SHAP Explainer
    explainer = shap.TreeExplainer(model)
    input_df = pd.DataFrame([user_input_dict])
    
    # Calculate SHAP values for the "Denied" class
    shap_values = explainer.shap_values(input_df)
    
    # Handle SHAP output format (can vary by version)
    if isinstance(shap_values, list):
        vals = shap_values[1][0] # Class 1 (Denied)
    else:
        vals = shap_values[0]

    # Map math values to feature names
    feature_importance = dict(zip(X.columns, vals))
    
    # Sort by highest impact (the reasons for denial)
    reasons = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    return reasons

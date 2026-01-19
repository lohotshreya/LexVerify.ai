import pandas as pd
import shap
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 1. Create a "Fake" Bank Model for the demo
def get_mock_model_and_data():
    data = pd.DataFrame({
        'income_monthly': [5000, 2000, 8000, 1500, 6000, 2200, 4500, 1200],
        'debt_amount': [200, 800, 100, 900, 300, 700, 400, 950],
        'credit_history_years': [10, 2, 15, 1, 8, 3, 7, 1],
        'denied': [0, 1, 0, 1, 0, 1, 0, 1] 
    })
    X = data.drop('denied', axis=1)
    y = data['denied']
    
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    return model, X

# 2. The XAI Engine: Fixed for NumPy compatibility
def explain_denial(user_input_dict):
    model, X = get_mock_model_and_data()
    explainer = shap.TreeExplainer(model)
    input_df = pd.DataFrame([user_input_dict])
    
    # Get raw SHAP values
    shap_results = explainer.shap_values(input_df)
    
    # FIX: Handle different SHAP output formats (list vs array)
    if isinstance(shap_results, list):
        # For binary classifiers, index 1 is the 'Denied' class
        vals = shap_results[1].flatten() 
    else:
        # If it's a single array, flatten it to 1D
        vals = shap_results.flatten()

    # CRITICAL FIX: Convert NumPy values to standard Python floats 
    # This prevents the "ambiguous truth value" error during sorting
    vals = [float(v) for v in vals]
    
    # Map names to values
    feature_importance = dict(zip(X.columns, vals))
    
    # Sort by highest impact
    reasons = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    return reasons

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

def get_predictions(model, X, model_type):
    if model_type == 'XGBoost':
        return model.predict(xgb.DMatrix(X))
    elif model_type == 'Neural Network':
        return model.predict(X).flatten()
    else:
        return model.predict(X)

# Load ensemble package
ensemble = joblib.load('final_zhvi_ensemble_model.pkl')

models = ensemble['base_models']
scaler = ensemble['scaler']
poly = ensemble['poly']
original_features = ensemble['feature_names']
meta_model = ensemble['meta_model']

# Define model MAPE scores for weighting
model_mape = {
    'Lasso': 0.041,
    'Ridge': 0.042,
    'Random Forest': 0.039,
    'XGBoost': 0.036,
    'Neural Network': 0.037
}
# Calculate inverse MAPE weights
model_weights = {model: 1/mape for model, mape in model_mape.items()}
total_weight = sum(model_weights.values())
normalized_weights = {model: weight/total_weight for model, weight in model_weights.items()}

# Streamlit UI
st.title("NYC ZHVI Prediction Tool 🏠📈")
st.write("Select a model and input your feature values to predict Zillow Housing Index (ZHVI)")
st.sidebar.header('Model and Input Settings')

# Expanded Model Selection
model_choice = st.sidebar.selectbox(
    'Select Model',
    list(models.keys()) + ['Stacking Ensemble', 'Simple Average Ensemble', 'Weighted Average Ensemble']
)

# Input fields for features
st.sidebar.subheader('Feature Inputs')

input_data = {}
# Year input
input_data['Year'] = st.sidebar.number_input('Year', min_value=1996, max_value=2020, step=1, value=2010)
# Month input
input_data['Month'] = st.sidebar.number_input('Month', min_value=1, max_value=12, step=1, value=1)

# Calculate TimeIndex automatically
input_data['TimeIndex'] = (input_data['Year'] - 1996) * 12 + (input_data['Month'] - 1)

# Other features
for feature in ['Unemployment Rate', 'CPI', 'Interest Rate', 'GDP Growth']:
    input_data[feature] = st.sidebar.number_input(f"{feature}", step=0.1)


# Convert input to DataFrame
input_df = pd.DataFrame([input_data])

# Preprocess input
input_poly = poly.transform(input_df)
input_poly = np.column_stack([np.ones(input_poly.shape[0]), input_poly])  # Add bias column
input_scaled = scaler.transform(input_poly)

# Prediction Logic
if st.button('Predict ZHVI'):
    if model_choice == 'Stacking Ensemble':
        # Stacking
        base_preds = []
        for model_name in models:
            model = models[model_name]
            pred = get_predictions(model, input_scaled, model_name)
            base_preds.append(pred.flatten())

        base_preds = np.column_stack(base_preds)
        prediction = meta_model.predict(base_preds)

    elif model_choice == 'Simple Average Ensemble':
        # Simple average
        preds = []
        for model_name in models:
            model = models[model_name]
            pred = get_predictions(model, input_scaled, model_name)
            preds.append(pred.flatten())
        prediction = np.mean(preds, axis=0)

    elif model_choice == 'Weighted Average Ensemble':
        # Weighted average
        preds = []
        weights = []
        for model_name in models:
            if model_name in model_weights:
                model = models[model_name]
                pred = get_predictions(model, input_scaled, model_name)
                preds.append(pred.flatten())
                weights.append(normalized_weights[model_name])

        preds = np.array(preds)
        weights = np.array(weights)
        prediction = np.average(preds, axis=0, weights=weights)

    else:
        # Individual model
        model = models[model_choice]
        prediction = get_predictions(model, input_scaled, model_choice)

    st.subheader(f"Predicted ZHVI: ${prediction[0]:,.2f}")

st.header("Plot of Economical Features Used As Input📈")
st.write("Use these features as a guideline for inputs. Dates range from 1996 to 2020.")

st.image("img/zhvi_cpi.png", 
         caption="ZHVI vs CPI Relationship")

st.image("img/zhvi_ir.png", 
         caption="ZHVI vs Interest Rate Relationship")

st.image("img/zhvi_unem.png", 
         caption="ZHVI vs Unemployment Rate Relationship")

st.image("img/zhvi_gdp.png", 
         caption="ZHVI vs GDP Growth Rate Relationship")

# Model Performance Comparison
st.header("Model Performance Overview 📊")

performance_data = {
    'Model': ['Weighted Average', 'Neural Network', 'Ridge', 'OLS', 'Lasso', 'Simple Average', 'XGBoost', 'Stacking Ensemble', 'Random Forest'],
    'MAE': [21694.88, 27119.78, 28891.17, 112483.28, 34030.37, 38783.54, 90674.54, 94035.58, 102800.77],
    'RMSE': [24867.89, 54902.67, 34587.00, 130532.10, 44618.02, 40336.11, 102249.80, 96290.63, 110934.40],
    'MAPE': [0.0369, 0.0447, 0.0482, 0.0536, 0.0549, 0.0647, 0.1449, 0.1544, 0.1708],
    'R2': [0.7498, 0.1694, 0.8469, -0.1439, 0.7853, 0.3417, -3.2301, -2.7514, -2.391]
}
performance_df = pd.DataFrame(performance_data)

st.dataframe(performance_df.sort_values('RMSE'))

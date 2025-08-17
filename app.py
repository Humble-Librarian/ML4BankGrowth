import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load model and data
model = joblib.load("ridge_model.pkl")   # adjust name if needed
X = pd.read_csv("selected_features.csv") # adjust name if needed
feature_names = X.columns

# --- Streamlit UI ---
st.set_page_config(page_title="Bank Lending Rate Predictor", layout="centered")

st.title("🏦 Bank Lending Rate Prediction")
st.write("This app predicts the **lending rate** of a bank based on regional and economic features using a Ridge Regression model.")

st.sidebar.header("🔧 Input Features")
user_inputs = {}
for feature in feature_names:
    user_inputs[feature] = st.sidebar.number_input(f"{feature}", value=0.0)

# Convert inputs into DataFrame
input_df = pd.DataFrame([user_inputs])

# Prediction
if st.sidebar.button("Predict Lending Rate"):
    prediction = model.predict(input_df)[0]
    
    st.subheader("📊 Prediction Result")
    st.success(f"Predicted Lending Rate: **{prediction:.2f}%**")
    st.caption("Model: Ridge Regression | R² ≈ 0.80")

    # --- Feature Importance (Top 5) ---
    st.subheader("📈 Top 5 Important Features")
    coef = model.coef_.flatten()
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": np.abs(coef)
    }).sort_values(by="importance", ascending=False).head(5)

    fig, ax = plt.subplots()
    ax.barh(importance_df["feature"], importance_df["importance"], color="skyblue")
    ax.set_xlabel("Coefficient Magnitude")
    ax.set_title("Top 5 Feature Importances")
    plt.gca().invert_yaxis()
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("📌 *Built with Streamlit & Scikit-learn | Ridge Regression*")

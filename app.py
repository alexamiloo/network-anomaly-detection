import streamlit as st
import pandas as pd
import preprocess
from tensorflow.keras.models import load_model
import numpy as np

st.title("Network Anomaly Detection System")

uploaded_file = st.file_uploader("Upload Network Flow CSV", type="csv")
if uploaded_file:
    model = load_model("models/autoencoder_model.h5")
    df = pd.read_csv(uploaded_file)
    df_clean = preprocess.load_and_clean_csv(uploaded_file)
    X_scaled, _, _ = preprocess.scale_features(df_clean)

    reconstructed = model.predict(X_scaled)
    mse = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)
    threshold = np.percentile(mse, 90)
    pred = (mse > threshold).astype(int)

    result_df = df.copy()
    result_df["AnomalyScore"] = mse
    result_df["Anomaly"] = ["Yes" if x else "No" for x in pred]

    st.write(result_df)
    st.success(f"Anomalies Detected: {np.sum(pred)}")

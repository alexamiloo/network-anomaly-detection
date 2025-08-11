import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

import preprocess

st.title("Network Anomaly Detection System")

uploaded_file = st.file_uploader("Upload Network Flow CSV", type="csv")
if uploaded_file:
    model = load_model("models/autoencoder_model.h5")
    scaler = preprocess.load_scaler("models/scaler.gz")

    # Read raw data for display and reset pointer for preprocessing
    df_raw = pd.read_csv(uploaded_file)
    uploaded_file.seek(0)
    X_scaled, _, _ = preprocess.preprocess_data(uploaded_file, scaler=scaler)

    reconstructed = model.predict(X_scaled, verbose=0)
    mse = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)
    threshold = np.percentile(mse, 90)
    pred = (mse > threshold).astype(int)

    result_df = df_raw.copy()
    result_df["AnomalyScore"] = mse
    result_df["Anomaly"] = ["Yes" if x else "No" for x in pred]

    st.write(result_df)
    st.success(f"Anomalies Detected: {np.sum(pred)}")

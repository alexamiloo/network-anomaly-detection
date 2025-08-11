"""Detection script using a trained autoencoder model."""

import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

import preprocess


def detect(data_path: str = "data/processed/cicids2017_test.csv",
           model_path: str = "models/autoencoder_model.h5",
           scaler_path: str = "models/scaler.gz") -> str:
    """Run anomaly detection on ``data_path``.

    Parameters
    ----------
    data_path: str
        Path to the CSV file containing data to score.
    model_path: str
        Path to the trained autoencoder model.
    scaler_path: str
        Path to the saved :class:`~sklearn.preprocessing.StandardScaler`.

    Returns
    -------
    str
        Path to the CSV file with detection results.
    """
    scaler = preprocess.load_scaler(scaler_path)
    X_test, y_test, _ = preprocess.preprocess_data(data_path, scaler=scaler)

    model = load_model(model_path)
    reconstructed = model.predict(X_test, verbose=0)
    mse = np.mean(np.power(X_test - reconstructed, 2), axis=1)

    threshold = np.percentile(mse, 90)
    predictions = (mse > threshold).astype(int)

    print(f"Threshold used: {threshold:.4f}")
    print(f"Detected anomalies: {np.sum(predictions)} / {len(predictions)}")

    result_df = pd.DataFrame(
        {
            "MSE": mse,
            "Prediction": predictions,
            "GroundTruth": y_test.values if y_test is not None else "Unknown",
        }
    )
    if not os.path.exists("evaluation"):
        os.makedirs("evaluation")
    output_path = "evaluation/detections.csv"
    result_df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    detect()

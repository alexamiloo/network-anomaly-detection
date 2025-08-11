from tensorflow.keras.models import load_model
import preprocess
import numpy as np
import pandas as pd

# Load data
X_test, y_test = preprocess.preprocess_data("data/processed/cicids2017_test.csv")

# Load trained model
model = load_model("models/autoencoder_model.h5")

# Predict reconstruction
reconstructed = model.predict(X_test)
mse = np.mean(np.power(X_test - reconstructed, 2), axis=1)

# Auto threshold (90th percentile)
threshold = np.percentile(mse, 90)
predictions = (mse > threshold).astype(int)

# Report
print(f"Threshold used: {threshold:.4f}")
print(f"Detected anomalies: {np.sum(predictions)} / {len(predictions)}")

# Save results
result_df = pd.DataFrame({
    "MSE": mse,
    "Prediction": predictions,
    "GroundTruth": y_test.values if y_test is not None else "Unknown"
})
result_df.to_csv("evaluation/detections.csv", index=False)

"""Training script for the network anomaly detection autoencoder."""

import os
from typing import Tuple

import preprocess
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential


def build_autoencoder(input_dim: int) -> Sequential:
    """Construct a simple feed-forward autoencoder."""
    model = Sequential(
        [
            Dense(64, activation="relu", input_shape=(input_dim,)),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(16, activation="relu"),
            Dense(32, activation="relu"),
            Dropout(0.2),
            Dense(64, activation="relu"),
            Dense(input_dim, activation="linear"),  # Output layer
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def train(data_path: str = "data/processed/cicids2017.csv") -> Tuple[str, str]:
    """Train the autoencoder model.

    Parameters
    ----------
    data_path: str
        Path to the preprocessed CSV file used for training.

    Returns
    -------
    tuple
        Paths to the saved model and scaler files.
    """
    X, _, scaler = preprocess.preprocess_data(data_path)
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

    model = build_autoencoder(X_train.shape[1])

    if not os.path.exists("models"):
        os.makedirs("models")

    early_stop = EarlyStopping(monitor="val_loss", patience=5)
    checkpoint_path = "models/autoencoder_model.h5"
    checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True)

    model.fit(
        X_train,
        X_train,
        validation_data=(X_val, X_val),
        epochs=100,
        batch_size=128,
        callbacks=[early_stop, checkpoint],
        verbose=0,
    )

    scaler_path = "models/scaler.gz"
    preprocess.save_scaler(scaler, scaler_path)
    return checkpoint_path, scaler_path


if __name__ == "__main__":
    train()

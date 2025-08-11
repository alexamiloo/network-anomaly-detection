"""Utility functions for preprocessing network flow data.

This module centralizes the preprocessing steps used across the training,
detection and Streamlit application scripts. In addition to cleaning and
encoding the raw data, it now supports persisting the feature scaler so the
same transformation used during training can be reapplied during inference.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_clean_csv(file_path):
    """
    Loads a CSV file, removes unnecessary columns, fills missing values.
    """
    df = pd.read_csv(file_path)

    drop_cols = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'Src IP', 'Dst IP']
    df.drop([col for col in drop_cols if col in df.columns], axis=1, inplace=True)

    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    return df

def encode_labels(df, label_col='Label'):
    """
    Convert attack labels to binary: 0 for normal, 1 for anomaly.
    """
    if label_col in df.columns:
        df[label_col] = df[label_col].apply(lambda x: 0 if 'BENIGN' in str(x).upper() else 1)
    return df

def scale_features(df, label_col="Label"):
    """Fit a :class:`~sklearn.preprocessing.StandardScaler` to ``df``.

    Parameters
    ----------
    df: pandas.DataFrame
        Cleaned dataframe containing numeric features and optionally a label
        column.
    label_col: str, default "Label"
        Name of the label column.

    Returns
    -------
    tuple
        ``(X_scaled, y, scaler)`` where ``scaler`` is the fitted
        ``StandardScaler`` instance.
    """
    y = df[label_col] if label_col in df.columns else None
    X = df.drop(columns=[label_col]) if label_col in df.columns else df

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler


def scale_features_with_scaler(df, scaler, label_col="Label"):
    """Scale ``df`` using a pre-fitted ``scaler``.

    This helper ensures that inference uses the exact same transformation as
    training by reusing the previously fitted scaler.
    """
    y = df[label_col] if label_col in df.columns else None
    X = df.drop(columns=[label_col]) if label_col in df.columns else df
    X_scaled = scaler.transform(X)
    return X_scaled, y


def save_scaler(scaler, path):
    """Persist ``scaler`` to ``path`` using :mod:`joblib`."""
    joblib.dump(scaler, path)


def load_scaler(path):
    """Load a scaler previously saved with :func:`save_scaler`."""
    return joblib.load(path)


def preprocess_data(file_path, scaler=None):
    """Full preprocessing pipeline.

    Parameters
    ----------
    file_path: str or buffer
        Path to a CSV file or a file-like object.
    scaler: Optional[StandardScaler]
        When provided, this scaler is used to transform the features instead
        of fitting a new one.
    """
    df = load_and_clean_csv(file_path)
    df = encode_labels(df)
    if scaler is not None:
        X_scaled, y = scale_features_with_scaler(df, scaler)
        return X_scaled, y, scaler
    return scale_features(df)

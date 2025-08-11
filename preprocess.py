import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

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

def scale_features(df, label_col='Label'):
    """
    Separates features and label, scales features using StandardScaler.
    """
    y = df[label_col] if label_col in df.columns else None
    X = df.drop(columns=[label_col]) if label_col in df.columns else df

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

def preprocess_data(file_path):
    df = load_and_clean_csv(file_path)
    df = encode_labels(df)
    X_scaled, y, _ = scale_features(df)
    return X_scaled, y

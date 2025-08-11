# Network Anomaly Detection

This project demonstrates a simple anomaly detection pipeline for network flow
records using an autoencoder neural network.

## Features

- Data cleaning, label encoding and feature scaling with persistent
  `StandardScaler`.
- Autoencoder model training with early stopping and model checkpoints.
- Command-line interface to train, detect anomalies and evaluate results.
- Streamlit web application for interactive anomaly scoring.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Train the model and save the scaler:

```bash
python main.py train --data path/to/train.csv
```

Run detection on new data:

```bash
python main.py detect --data path/to/test.csv
```

Evaluate stored detections:

```bash
python main.py evaluate --results evaluation/detections.csv
```

To launch the Streamlit interface:

```bash
streamlit run app.py
```

## Data

The scripts expect CSV files formatted similarly to the CICIDS2017 dataset. See
`preprocess.py` for details on required columns.

## License

MIT

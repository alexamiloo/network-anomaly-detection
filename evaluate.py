"""Evaluation utilities for detection results."""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import seaborn as sns


def evaluate(results_path: str = "evaluation/detections.csv") -> None:
    """Generate evaluation metrics and plots from detection results."""
    df = pd.read_csv(results_path)
    y_true = df["GroundTruth"]
    y_pred = df["Prediction"]
    mse = df["MSE"]

    print("Classification Report:\n", classification_report(y_true, y_pred))

    if not os.path.exists(os.path.dirname(results_path)):
        os.makedirs(os.path.dirname(results_path), exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("evaluation/confusion_matrix.png")
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, mse)
    auc = roc_auc_score(y_true, mse)

    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig("evaluation/roc_curve.png")
    plt.close()


if __name__ == "__main__":
    evaluate()

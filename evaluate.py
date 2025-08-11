import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
import seaborn as sns

df = pd.read_csv("evaluation/detections.csv")

y_true = df["GroundTruth"]
y_pred = df["Prediction"]
mse = df["MSE"]

# Metrics
print("Classification Report:\n", classification_report(y_true, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("evaluation/confusion_matrix.png")
plt.close()

# ROC Curve
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

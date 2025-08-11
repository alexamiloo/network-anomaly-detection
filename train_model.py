from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import preprocess
import os

# Load and preprocess data
X, y = preprocess.preprocess_data("data/processed/cicids2017.csv")
X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

# Build autoencoder model
def build_autoencoder(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(input_dim, activation='linear')  # Output layer
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

model = build_autoencoder(X_train.shape[1])

# Callbacks
if not os.path.exists("models"):
    os.makedirs("models")

early_stop = EarlyStopping(monitor="val_loss", patience=5)
checkpoint = ModelCheckpoint("models/autoencoder_model.h5", save_best_only=True)

# Train
model.fit(
    X_train, X_train,
    validation_data=(X_val, X_val),
    epochs=100,
    batch_size=128,
    callbacks=[early_stop, checkpoint]
)

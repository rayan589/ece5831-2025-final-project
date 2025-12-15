import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

os.makedirs("state", exist_ok=True)
os.makedirs("charts", exist_ok=True)

DATA_PATH    = "data/metrics_multi.csv"
BASE_WEIGHTS = "state/lstm_base.weights.h5"
BASE_SCALER  = "state/lstm_base_scaler.npz"
LSTM_WINDOW  = 30

def build_lstm_model(timesteps, num_feats):
    model = Sequential([
        LSTM(32, input_shape=(timesteps, num_feats)),
        Dense(num_feats)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def make_sequences(arr, window=LSTM_WINDOW):
    X, y = [], []
    T = len(arr)
    for i in range(T - window):
        X.append(arr[i:i+window])
        y.append(arr[i+window])
    return np.array(X, dtype="float32"), np.array(y, dtype="float32")

def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values(["site_id", "date"])

    feats = df[["sessions", "conversion_rate", "revenue"]].values.astype("float32")

    # Global MinMax scaling
    f_min = feats.min(axis=0)
    f_max = feats.max(axis=0)
    denom = (f_max - f_min)
    denom[denom == 0.0] = 1e-6
    feats_scaled = (feats - f_min) / denom

    np.savez(BASE_SCALER, f_min=f_min, f_max=f_max)

    X, y = make_sequences(feats_scaled, window=LSTM_WINDOW)
    print("Pretraining data shapes:", X.shape, y.shape)

    model = build_lstm_model(LSTM_WINDOW, feats.shape[1])

    # Train/val split
    N = len(X)
    split = int(0.9 * N)
    X_train, y_train = X[:split], y[:split]
    X_val,   y_val   = X[split:], y[split:]

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val) if len(X_val) > 0 else None,
        epochs=80,              # allow more epochs, ES will cut early
        batch_size=64,          # slightly larger batch size for more data
        callbacks=[early_stop],
        verbose=1
    )

    model.save_weights(BASE_WEIGHTS)
    print(f"Saved base LSTM weights to {BASE_WEIGHTS}")

    # Plot training vs validation loss
    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Base LSTM Pretraining Loss")
    plt.legend()
    plt.tight_layout()
    out_chart = "charts/lstm_base_loss.png"
    plt.savefig(out_chart)
    plt.close()
    print(f"Saved base LSTM loss curve to {out_chart}")

if __name__ == "__main__":
    main()


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

os.makedirs("state", exist_ok=True)
os.makedirs("charts", exist_ok=True)

DATA_PATH   = "data/metrics_multi.csv"
BASE_WEIGHTS = "state/lstm_base.weights.h5"
FT_WEIGHTS   = "state/lstm_site0_finetuned.weights.h5"
FT_SCALER    = "state/lstm_site0_scaler.npz"
LSTM_WINDOW  = 30

def build_lstm_model(timesteps, num_feats):
    model = Sequential([
        LSTM(32, input_shape=(timesteps, num_feats)),
        Dense(num_feats)
    ])
    # Smaller LR for fine-tuning
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
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
    df_site = df[df["site_id"] == 0].sort_values("date")

    feats = df_site[["sessions", "conversion_rate", "revenue"]].values.astype("float32")

    # Site-specific MinMax scaling
    f_min = feats.min(axis=0)
    f_max = feats.max(axis=0)
    denom = (f_max - f_min)
    denom[denom == 0.0] = 1e-6
    feats_scaled = (feats - f_min) / denom

    np.savez(FT_SCALER, f_min=f_min, f_max=f_max)

    X, y = make_sequences(feats_scaled, window=LSTM_WINDOW)
    print("Fine-tune data shapes:", X.shape, y.shape)

    if len(X) < 10:
        print("Not enough data to fine-tune.")
        return

    model = build_lstm_model(LSTM_WINDOW, feats.shape[1])

    # Load base weights
    model.load_weights(BASE_WEIGHTS)
    print("Loaded base LSTM weights for fine-tuning.")

    N = len(X)
    split = int(0.8 * N)
    X_train, y_train = X[:split], y[:split]
    X_val,   y_val   = X[split:], y[split:]

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val) if len(X_val) > 0 else None,
        epochs=40,
        batch_size=16,
        callbacks=[early_stop],
        verbose=1
    )

    model.save_weights(FT_WEIGHTS)
    print(f"Saved fine-tuned site-specific weights to {FT_WEIGHTS}")
    print(f"Saved site-specific scaler to {FT_SCALER}")

    # Optional: plot fine-tune loss
    plt.figure()
    plt.plot(history.history["loss"], label="Fine-tune Train Loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Fine-tune Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Site 0 Fine-tuning Loss")
    plt.legend()
    plt.tight_layout()
    out_chart = "charts/lstm_site0_finetune_loss.png"
    plt.savefig(out_chart)
    plt.close()
    print(f"Saved fine-tune loss curve to {out_chart}")

if __name__ == "__main__":
    main()

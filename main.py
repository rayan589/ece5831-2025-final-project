import os, json
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# paths + constants
DATA_PATH = "data/metrics.csv"
MEM_PATH  = "state/memory.json"
CHART     = "charts/conv_rate.png"

FT_WEIGHTS   = "state/lstm_site0_finetuned.weights.h5"
FT_SCALER    = "state/lstm_site0_scaler.npz"
LSTM_WINDOW  = 30

ACTIONS_PLAYBOOK = [
    ("conv_drop",   "Run A/B on landing-page headline (target +10% CVR)."),
    ("traffic_drop","Shift posting time by +2h for 48h; compare sessions."),
    ("rev_drop",    "Send one-time email resend to non-openers at +24h."),
    ("spike",       "Replicate winning channel creative; +10% budget for 2 days."),
    ("stable",      "No action; monitor another day.")
]

#FOlders and memory

def ensure_dirs():
    os.makedirs("state", exist_ok=True)
    os.makedirs("charts", exist_ok=True)

def load_memory():
    if not os.path.exists(MEM_PATH):
        return {"accepted": [], "rejected": []}
    with open(MEM_PATH) as f:
        return json.load(f)

def save_memory(mem):
    with open(MEM_PATH, "w") as f:
        json.dump(mem, f, indent=2)


def stl_z(series, period=7):
    """Return robust z-score of the latest residual vs historical residuals."""
    y = pd.Series(series).astype(float)
    res = STL(y, period=period, robust=True).fit()
    resid = res.resid
    hist = resid.iloc[:-1]
    med  = np.median(hist)
    mad  = np.median(np.abs(hist - med)) or 1e-9
    z = (resid.iloc[-1] - med) / (1.4826 * mad)
    return float(z)


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

def compute_lstm_zscores(df):
    feats = df[["sessions", "conversion_rate", "revenue"]].values.astype("float32")
    T, F = feats.shape
    if T <= LSTM_WINDOW + 5:
        return {"sessions": 0.0, "conv_rate": 0.0, "revenue": 0.0}

    if not (os.path.exists(FT_WEIGHTS) and os.path.exists(FT_SCALER)):
        print("Fine-tuned weights or scaler not found, returning zeros for LSTM z-scores.")
        return {"sessions": 0.0, "conv_rate": 0.0, "revenue": 0.0}

    # Load scaler
    scal = np.load(FT_SCALER)
    f_min = scal["f_min"]
    f_max = scal["f_max"]
    denom = (f_max - f_min)
    denom[denom == 0.0] = 1e-6
    feats_scaled = (feats - f_min) / denom

    X, y = make_sequences(feats_scaled, window=LSTM_WINDOW)

    model = build_lstm_model(LSTM_WINDOW, F)
    model.load_weights(FT_WEIGHTS)

    y_pred = model.predict(X, verbose=0)
    errors = np.abs(y_pred - y)  # [N, 3]

    zscores = {}
    metric_keys = ["sessions", "conv_rate", "revenue"]
    for j, key in enumerate(metric_keys):
        e = errors[:, j]
        hist = e[:-1]
        last = e[-1]
        med  = np.median(hist)
        mad  = np.median(np.abs(hist - med)) or 1e-9
        z = (last - med) / (1.4826 * mad)
        zscores[key] = float(z)

    return zscores

#Hybrid anomaly detection

def detect_anomalies(df):
    low_volume = df["sessions"].iloc[-1] < 300

    stats_z = {
        "sessions":  stl_z(df["sessions"]),
        "revenue":   stl_z(df["revenue"]),
        "conv_rate": stl_z(df["conversion_rate"])
    }

    lstm_z = compute_lstm_zscores(df)

    hybrid_z = {}
    for k in stats_z.keys():
        hybrid_z[k] = 0.5 * stats_z[k] + 0.5 * lstm_z.get(k, 0.0)

    levels = {}
    for k, z in hybrid_z.items():
        lvl = "none"
        az = abs(z)
        if az >= 3.5:
            lvl = "red"
        elif az >= 2.5:
            lvl = "yellow"
        levels[k] = (lvl, z)

    if low_volume:
        levels = {k: ("none", z) for k, (_, z) in levels.items()}

    return levels, stats_z, lstm_z

#Agentic playbook decision

def choose_action(df, levels, memory):
    prev, today = df.iloc[-2], df.iloc[-1]
    ideas = []

    # prioritize red > yellow, then larger |z|
    order = sorted(
        levels.items(),
        key=lambda kv: (kv[1][0] != "red", kv[1][0] != "yellow", -abs(kv[1][1]))
    )

    for metric, (lvl, z) in order:
        if lvl == "none":
            continue
        if metric == "conv_rate" and z < 0:
            ideas.append(("conv_drop", ACTIONS_PLAYBOOK[0][1]))
        elif metric == "sessions" and z < 0:
            ideas.append(("traffic_drop", ACTIONS_PLAYBOOK[1][1]))
        elif metric == "revenue" and z < 0:
            if abs(stl_z(df["conversion_rate"])) < 2.0:
                ideas.append(("traffic_drop", ACTIONS_PLAYBOOK[1][1]))
            else:
                ideas.append(("rev_drop", ACTIONS_PLAYBOOK[2][1]))
        elif z > 0:
            ideas.append(("spike", ACTIONS_PLAYBOOK[3][1]))

    if not ideas:
        ideas = [("stable", ACTIONS_PLAYBOOK[4][1])]

    # avoid recently rejected ideas
    rejected = set(memory.get("rejected", []))
    for key, txt in ideas:
        if txt not in rejected:
            return key, txt

    return ideas[0]

#Plotting

def plot_conv(df):
    plt.figure()
    plt.plot(df["date"], df["conversion_rate"])
    plt.title("Conversion Rate (last 60 days)")
    plt.xlabel("Date")
    plt.ylabel("Rate")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(CHART)
    plt.close()

#main daily loop

def main():
    ensure_dirs()
    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date")
    mem = load_memory()

    levels, stats_z, lstm_z = detect_anomalies(df)
    key, action = choose_action(df, levels, mem)
    plot_conv(df.tail(60))

    latest = df.iloc[-1]
    msg = f"""
Daily Brief — {datetime.now().date()}
KPI today:
- sessions: {int(latest['sessions'])}
- conversion_rate: {latest['conversion_rate']:.3f}
- revenue: ${latest['revenue']:.2f}

Anomalies (Hybrid z-scores: STL + LSTM):
- sessions: {levels['sessions'][0]} (z={levels['sessions'][1]:+.2f}, stl={stats_z['sessions']:+.2f}, lstm={lstm_z['sessions']:+.2f})
- conv_rate: {levels['conv_rate'][0]} (z={levels['conv_rate'][1]:+.2f}, stl={stats_z['conv_rate']:+.2f}, lstm={lstm_z['conv_rate']:+.2f})
- revenue:  {levels['revenue'][0]} (z={levels['revenue'][1]:+.2f}, stl={stats_z['revenue']:+.2f}, lstm={lstm_z['revenue']:+.2f})

Suggested next step:
→ {action}

Chart saved: {CHART}
Approve action? (y/n): """
    ans = input(msg).strip().lower()
    if ans == "y":
        print("Logged as approved.")
        mem.setdefault("accepted", []).append(action)
    else:
        print("Logged as rejected. I’ll deprioritize it next time.")
        mem.setdefault("rejected", []).append(action)
    save_memory(mem)

if __name__ == "__main__":
    main()

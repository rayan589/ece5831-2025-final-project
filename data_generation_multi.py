import os
import numpy as np
import pandas as pd

np.random.seed(42)

os.makedirs("data", exist_ok=True)

# More sites = more data
N_SITES = 50        
DAYS = 180
START_DATE = "2025-01-01"

all_rows = []

for site_id in range(N_SITES):
    dates = pd.date_range(START_DATE, periods=DAYS)

  
    base_level = np.random.uniform(1000, 2000)
    noise_scale = np.random.uniform(150, 350)

    trend = np.linspace(1.0, np.random.uniform(1.05, 1.25), DAYS)
    base_sessions = np.random.normal(base_level, noise_scale, DAYS) * trend

    weekday_adj = np.where(pd.Series(dates).dt.dayofweek >= 5, 0.8, 1.0)
    sessions = base_sessions * weekday_adj

    conv_rate = np.clip(np.random.normal(0.022, 0.003, DAYS), 0.015, 0.035)
    aov = np.random.uniform(40, 80, DAYS)

    # anomalies per site
    idxs = np.random.choice(range(DAYS), 5, replace=False)
    anomaly_flags = np.zeros(DAYS, dtype=int)

    drop_idx = idxs[:2]
    spike_idx = idxs[2:]

    sessions[drop_idx] *= 0.5
    sessions[spike_idx] *= 1.6
    anomaly_flags[drop_idx] = 1
    anomaly_flags[spike_idx] = 1

    sessions = np.rint(sessions).astype(int)
    revenue = sessions * conv_rate * aov

    df_site = pd.DataFrame({
        "site_id": site_id,
        "date": dates,
        "sessions": sessions,
        "conversion_rate": np.round(conv_rate, 3),
        "revenue": np.round(revenue, 2),
        "is_anomaly": anomaly_flags
    })
    all_rows.append(df_site)

df_all = pd.concat(all_rows, ignore_index=True)

multi_path = os.path.join("data", "metrics_multi.csv")
df_all.to_csv(multi_path, index=False)
print(f"Saved multi-site KPI dataset to {multi_path}")

# Target site 0 for your copilot / fine-tuning
site0 = df_all[df_all["site_id"] == 0].copy()
single_path = os.path.join("data", "metrics.csv")
site0.to_csv(single_path, index=False)
print(f"Saved target site 0 metrics to {single_path}")

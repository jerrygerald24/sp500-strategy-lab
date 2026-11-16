import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
FULL_PKL = BASE_DIR / "data" / "sp500_full_matrix.pkl"

print("Loading full 2000-2026 matrix...")
df = pd.read_pickle(FULL_PKL)
dates = df.index
prices = df.to_numpy(dtype=np.float64)
n_days, n_stocks = prices.shape

ret_1d = df.pct_change(1).to_numpy()
ret_5d = df.pct_change(5).to_numpy()
ret_20d = df.pct_change(20).to_numpy()
vol_20d = df.pct_change(1).rolling(20).std().to_numpy()
sma50 = df.rolling(50).mean().to_numpy()
sma200 = df.rolling(200).mean().to_numpy()
dist_sma50 = (prices - sma50) / sma50
dist_sma200 = (prices - sma200) / sma200

fwd_20d = (df.shift(-20) / df - 1.0).to_numpy()
mkt_fwd_20d = np.nanmedian(fwd_20d, axis=1, keepdims=True)
y_all = (fwd_20d > mkt_fwd_20d).astype(np.float32)

# Train on 2000-2018 (18 years!)
train_mask_days = (dates >= "2000-11-01") & (dates < "2019-01-01")
X_train_list, y_train_list = [], []
for t in np.where(train_mask_days)[0][::5]:
    valid = ~np.isnan(ret_1d[t]) & ~np.isnan(dist_sma200[t]) & ~np.isnan(vol_20d[t]) & ~np.isnan(fwd_20d[t])
    if np.sum(valid) < 50:
        continue
    X_train_list.append(np.column_stack([
        ret_1d[t, valid], ret_5d[t, valid], ret_20d[t, valid],
        vol_20d[t, valid], dist_sma50[t, valid], dist_sma200[t, valid]
    ]))
    y_train_list.append(y_all[t, valid])

X_train = np.vstack(X_train_list)
y_train = np.concatenate(y_train_list)
print(f"Training HistGradientBoostingClassifier on {len(X_train):,} samples (2000-2018)...")

model = HistGradientBoostingClassifier(max_iter=150, max_depth=4, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

iso_weeks = dates.isocalendar().week.to_numpy()

def eval_window(label, start_dt, end_dt):
    print(f"\n=== {label} ({start_dt} to {end_dt}) ===")
    mask = (dates >= start_dt) & (dates <= end_dt)
    indices = np.where(mask)[0]
    end_idx = indices[-1]
    for mode in ["benchmark", "raw_bottom5", "ml_dip5"]:
        shares = np.zeros(n_stocks)
        contributed = 0.0
        for t in indices:
            if iso_weeks[t] != iso_weeks[t - 1]:
                p_now = prices[t]
                valid = ~np.isnan(p_now) & (p_now > 0) & ~np.isnan(ret_1d[t]) & ~np.isnan(dist_sma200[t])
                idx = np.where(valid)[0]
                if len(idx) < 20:
                    continue
                if mode == "benchmark":
                    shares[idx] += (482.0 / len(idx)) / p_now[idx]
                elif mode == "raw_bottom5":
                    chosen = idx[np.argpartition(ret_1d[t, idx], 5)[:5]]
                    shares[chosen] += (482.0 / 5) / p_now[chosen]
                elif mode == "ml_dip5":
                    losers20 = idx[np.argpartition(ret_1d[t, idx], 20)[:20]]
                    X_cand = np.column_stack([
                        ret_1d[t, losers20], ret_5d[t, losers20], ret_20d[t, losers20],
                        vol_20d[t, losers20], dist_sma50[t, losers20], dist_sma200[t, losers20]
                    ])
                    probs = model.predict_proba(X_cand)[:, 1]
                    chosen = losers20[np.argpartition(probs, -5)[-5:]]
                    shares[chosen] += (482.0 / 5) / p_now[chosen]
                contributed += 482.0
        final_val = float(np.dot(shares, np.nan_to_num(prices[end_idx], nan=0.0)))
        roi = (final_val - contributed) / contributed * 100.0
        print(f"  {mode:12s} | Invested: ${contributed:,.0f} | Final: ${final_val:,.2f} | ROI: {roi:.2f}%")

eval_window("Out-of-Sample 1", "2019-01-01", "2023-09-29")
eval_window("Out-of-Sample 2 (Live Yahoo 2023-2026)", "2023-10-01", "2026-11-13")

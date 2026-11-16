import pandas as pd
import yfinance as yf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_CSV = BASE_DIR / "data" / "sp500_close.csv"
FULL_PKL = BASE_DIR / "data" / "sp500_full_matrix.pkl"

print("1. Building 2000-2023 matrix from local CSV...")
df = pd.read_csv(RAW_CSV, usecols=["ShareName", "Date", "Value"])
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
df = df.dropna(subset=["Value"])

hist = df.pivot_table(index="Date", columns="ShareName", values="Value")
hist.index = pd.to_datetime(hist.index)
hist = hist.sort_index()
# Keep tickers that have at least 500 days of history
hist = hist.dropna(axis=1, thresh=500)
tickers = list(hist.columns)
print(f"Historical matrix (2000-2023): {hist.shape[0]} days x {len(tickers)} stocks")

print("2. Fetching 2023-09-30 to today from Yahoo Finance...")
# Download in a single batch
yf_data = yf.download(tickers, start="2023-09-30", progress=False, threads=True)
if isinstance(yf_data.columns, pd.MultiIndex):
    recent = yf_data["Close"]
else:
    recent = yf_data

recent.index = pd.to_datetime(recent.index).tz_localize(None)
# Align columns
common_cols = [c for c in tickers if c in recent.columns]
combined = pd.concat([hist[common_cols], recent[common_cols]])
combined = combined[~combined.index.duplicated(keep="last")].sort_index().ffill()

combined.to_pickle(FULL_PKL)
print(f"Saved full 2000-2026 matrix: {combined.shape[0]} days x {combined.shape[1]} stocks ({combined.index[0].date()} to {combined.index[-1].date()})")

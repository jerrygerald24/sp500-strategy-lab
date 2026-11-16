import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MATRIX_PKL = BASE_DIR / "data" / "sp500_10yr_matrix.pkl"

app = FastAPI(title="S&P 500 Strategy Lab")

# Global cached price matrix
PRICE_DF = pd.read_pickle(MATRIX_PKL)
DATES = PRICE_DF.index
PRICES = PRICE_DF.to_numpy(dtype=np.float64)
TICKERS = PRICE_DF.columns.to_numpy()

class StrategyConfig(BaseModel):
    cadence: str = "daily"      # "daily" or "weekly"
    direction: str = "top"      # "top" (gainers) or "bottom" (losers) or "benchmark" (equal-weight S&P 500)
    top_n: int = 1              # 1, 5, 10
    lookback: int = 1           # 1, 5, 20, 60 trading days
    amount: float = 100.0       # $ per contribution

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return (BASE_DIR / "index.html").read_text(encoding="utf-8")

@app.post("/api/simulate")
def simulate(cfg: StrategyConfig):
    n_days, n_stocks = PRICES.shape
    lookback = max(1, cfg.lookback)

    # Precompute lookback returns matrix
    # ret[t] = (PRICES[t] - PRICES[t - lookback]) / PRICES[t - lookback]
    shares = np.zeros(n_stocks, dtype=np.float64)
    contributed = 0.0

    # Determine contribution day indices
    if cfg.cadence == "weekly":
        # First trading day of each ISO week
        iso_weeks = DATES.isocalendar().week.to_numpy()
        contrib_mask = np.zeros(n_days, dtype=bool)
        contrib_mask[lookback:] = True
        week_change = np.empty(n_days, dtype=bool)
        week_change[0] = True
        week_change[1:] = iso_weeks[1:] != iso_weeks[:-1]
        contrib_mask = contrib_mask & week_change
    elif cfg.cadence == "monthly":
        # First trading day of each month
        months = DATES.month.to_numpy()
        contrib_mask = np.zeros(n_days, dtype=bool)
        contrib_mask[lookback:] = True
        month_change = np.empty(n_days, dtype=bool)
        month_change[0] = True
        month_change[1:] = months[1:] != months[:-1]
        contrib_mask = contrib_mask & month_change
    else:
        contrib_mask = np.zeros(n_days, dtype=bool)
        contrib_mask[lookback:] = True

    equity_curve = np.zeros(n_days, dtype=np.float64)
    contrib_curve = np.zeros(n_days, dtype=np.float64)

    for t in range(lookback, n_days):
        if contrib_mask[t]:
            p_now = PRICES[t]
            if cfg.direction == "benchmark":
                valid = ~np.isnan(p_now) & (p_now > 0)
                if np.any(valid):
                    alloc = cfg.amount / np.sum(valid)
                    shares[valid] += alloc / p_now[valid]
                    contributed += cfg.amount
            else:
                p_prev = PRICES[t - lookback]
                valid = ~np.isnan(p_now) & ~np.isnan(p_prev) & (p_prev > 0) & (p_now > 0)
                if np.any(valid):
                    rets = np.where(valid, (p_now - p_prev) / p_prev, np.nan)
                    valid_idx = np.where(valid)[0]
                    valid_rets = rets[valid_idx]
                    k = min(cfg.top_n, len(valid_idx))
                    if cfg.direction == "top":
                        chosen = valid_idx[np.argpartition(valid_rets, -k)[-k:]]
                    else:
                        chosen = valid_idx[np.argpartition(valid_rets, k)[:k]]
                    alloc = cfg.amount / k
                    shares[chosen] += alloc / p_now[chosen]
                    contributed += cfg.amount

        # Daily portfolio value
        p_t = np.nan_to_num(PRICES[t], nan=0.0)
        equity_curve[t] = np.dot(shares, p_t)
        contrib_curve[t] = contributed

    # Slice from first contribution day
    start_idx = lookback
    eq = equity_curve[start_idx:]
    co = contrib_curve[start_idx:]
    dt = DATES[start_idx:]

    # Metrics
    final_val = float(eq[-1])
    total_inv = float(co[-1])
    profit = final_val - total_inv
    roi_pct = (profit / total_inv * 100.0) if total_inv > 0 else 0.0
    years = (dt[-1] - dt[0]).days / 365.25
    cagr_pct = (((final_val / total_inv) ** (1.0 / years)) - 1.0) * 100.0 if (total_inv > 0 and years > 0 and final_val > 0) else 0.0

    # Max drawdown of equity curve
    running_max = np.maximum.accumulate(eq)
    dd = np.where(running_max > 0, (eq - running_max) / running_max * 100.0, 0.0)
    max_dd = float(np.min(dd))

    # Downsample to ~weekly points for crisp, fast Chart.js rendering
    step = max(1, len(dt) // 250)
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in dt[::step]],
        "equity": [round(float(x), 2) for x in eq[::step]],
        "contributed": [round(float(x), 2) for x in co[::step]],
        "metrics": {
            "final_value": round(final_val, 2),
            "total_contributed": round(total_inv, 2),
            "profit": round(profit, 2),
            "roi_pct": round(roi_pct, 2),
            "cagr_pct": round(cagr_pct, 2),
            "max_drawdown_pct": round(max_dd, 2),
        }
    }

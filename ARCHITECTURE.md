# S&P 500 Strategy Lab — Architecture & Roadmap

## 1. Core Philosophy
**Heuristic Discovery → Visual Comparison → ML Fine-Tuning**
- Keep all heavy data processing, matrix math, and strategy simulation in **Python** (FastAPI + Pandas/NumPy).
- Keep the **UI** lightweight (HTML + Chart.js) so it only sends configuration JSON and renders returned curves.

---

## 2. Phase 1: Configurable Strategy Lab (POC → Full)
### Strategy Levers
- **Cadence:** Daily (`$100/day` default) vs. Weekly (`$100/week` or `$500/week` normalized)
- **Lookback Window:** `1d`, `5d`, `20d`, `60d` % return
- **Selection Rule:** Top $N$ (Momentum) vs. Bottom $N$ (Contrarian / Dip-Buying)
- **Basket Size ($N$):** `1` ($100/stock), `5` ($20/stock), `10` ($10/stock)

### Visuals & Metrics
- **Multi-Curve Overlay Chart:** Plot multiple strategy configurations simultaneously against **Total Cash Contributed** and **S&P 500 Equal-Weight / SPY Benchmark**.
- **Toggle:** Dollar Value (`$`) vs. ROI (`%`).
- **KPIs:** Final Value, Total Contributed, Net Profit, ROI %, CAGR %, Max Drawdown %.

---

## 3. Phase 2: ML Fine-Tuning (Future)
Once a promising heuristic pattern is identified:
1. **Feature Engineering:** `1d/5d/20d` returns, `20d` volatility, distance from `50d/200d` SMA.
2. **Model (XGBoost / LightGBM):** Predict next-period outperformance to filter out "value traps" and dynamically weight positions.
3. **Walk-Forward Validation:** Train on rolling past windows (e.g., 2013–2018) and test strictly on unseen future years (e.g., 2019–2023) to prevent overfitting.

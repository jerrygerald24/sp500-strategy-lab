import csv
import io
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
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
    weekday: int = 0            # 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri
    trailing_stop_pct: float = 0.0  # 0 = Never Sell; >0 = sell lot when it drops X% from post-buy max price
    start_date: str | None = None   # Optional YYYY-MM-DD
    end_date: str | None = None     # Optional YYYY-MM-DD
    selection_model: str = "all"    # "all", "low_vol_50", "low_vol_20"

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return (BASE_DIR / "index.html").read_text(encoding="utf-8")

@app.get("/sell-strategy", response_class=HTMLResponse)
def serve_sell_strategy():
    return (BASE_DIR / "sell_strategy.html").read_text(encoding="utf-8")

@app.get("/portfolio", response_class=HTMLResponse)
def serve_portfolio():
    return (BASE_DIR / "portfolio.html").read_text(encoding="utf-8")

@app.get("/api/portfolio")
def api_portfolio():
    import sqlite3
    db_path = BASE_DIR / "portfolio.db"
    if not db_path.exists():
        return {"summary": {"total_invested": 0, "current_value": 0, "profit": 0, "roi_pct": 0, "open_count": 0, "sold_count": 0}, "lots": [], "transactions": []}
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    lots = [dict(r) for r in conn.execute("SELECT * FROM lots ORDER BY status ASC, appreciation_pct DESC").fetchall()]
    txs = [dict(r) for r in conn.execute("SELECT * FROM transactions ORDER BY id DESC").fetchall()]
    conn.close()

    total_invested = sum(l["shares"] * l["buy_price"] for l in lots)
    open_val = sum(l["shares"] * l["current_price"] for l in lots if l["status"] == "OPEN")
    sold_val = sum(l["shares"] * (l["sell_price"] or 0) for l in lots if l["status"] == "SOLD")
    current_value = open_val + sold_val
    profit = current_value - total_invested
    roi_pct = (profit / total_invested * 100.0) if total_invested > 0 else 0.0
    return {
        "summary": {
            "total_invested": round(total_invested, 2),
            "current_value": round(current_value, 2),
            "open_value": round(open_val, 2),
            "sold_proceeds": round(sold_val, 2),
            "profit": round(profit, 2),
            "roi_pct": round(roi_pct, 2),
            "open_count": sum(1 for l in lots if l["status"] == "OPEN"),
            "sold_count": sum(1 for l in lots if l["status"] == "SOLD"),
        },
        "lots": lots,
        "transactions": txs,
    }

SELL_STRATEGY_CACHE = None

@app.get("/api/sell-strategy")
def api_sell_strategy():
    global SELL_STRATEGY_CACHE
    if SELL_STRATEGY_CACHE is None:
        from sell_strategy import run_stop_loss_backtest
        SELL_STRATEGY_CACHE = run_stop_loss_backtest(PRICE_DF)
    return SELL_STRATEGY_CACHE

class StrategyExportItem(BaseModel):
    name: str
    config: StrategyConfig

@app.post("/api/simulate")
def simulate(cfg: StrategyConfig):
    return _run_sim(cfg, collect_trades=False)

def _run_sim(cfg: StrategyConfig, collect_trades: bool = False):
    mask = np.ones(len(DATES), dtype=bool)
    if cfg.start_date:
        mask &= DATES >= pd.Timestamp(cfg.start_date)
    if cfg.end_date:
        mask &= DATES <= pd.Timestamp(cfg.end_date)
    dates = DATES[mask]
    prices = PRICES[mask]
    n_days, n_stocks = prices.shape
    lookback = max(1, cfg.lookback)
    if n_days <= lookback:
        return {
            "dates": [], "equity": [], "contributed": [],
            "metrics": {"final_value": 0, "holdings_value": 0, "spare_cash": 0, "lots_sold": 0, "total_contributed": 0, "profit": 0, "roi_pct": 0, "cagr_pct": 0, "max_drawdown_pct": 0},
            "transactions": [], "lots": []
        }

    # Precompute lookback returns matrix
    # ret[t] = (PRICES[t] - PRICES[t - lookback]) / PRICES[t - lookback]
    shares = np.zeros(n_stocks, dtype=np.float64)
    contributed = 0.0

    # Determine contribution day indices
    if cfg.cadence == "weekly":
        # First trading day on or after target weekday (0=Mon..4=Fri) in each ISO week
        iso_weeks = dates.isocalendar().week.to_numpy()
        iso_years = dates.isocalendar().year.to_numpy()
        weekdays = dates.weekday.to_numpy()
        contrib_mask = np.zeros(n_days, dtype=bool)
        seen_weeks = set()
        for t in range(lookback, n_days):
            yw = (int(iso_years[t]), int(iso_weeks[t]))
            if yw not in seen_weeks and weekdays[t] >= cfg.weekday:
                contrib_mask[t] = True
                seen_weeks.add(yw)
    elif cfg.cadence == "monthly":
        # First trading day of each month
        months = dates.month.to_numpy()
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
    spare_cash = 0.0
    daily_cash_rate = 1.03 ** (1.0 / 252.0)
    lots_sold = 0
    use_trailing = cfg.trailing_stop_pct > 0
    track_lots = use_trailing or collect_trades
    sl_frac = cfg.trailing_stop_pct / 100.0
    # Each lot: [ticker_idx, shares, max_price, buy_date_str, buy_price]
    lots = []
    transactions = []
    closed_lots = []

    for t in range(lookback, n_days):
        p_t = np.nan_to_num(prices[t], nan=0.0)
        d_str = dates[t].strftime("%Y-%m-%d") if collect_trades else ""

        # Compound spare cash at 3% YoY
        if spare_cash > 0:
            spare_cash *= daily_cash_rate

        # Check trailing stop-loss from post-buy max price
        if use_trailing and lots:
            running_holdings = sum(l[1] * p_t[l[0]] for l in lots) if collect_trades else 0.0
            remaining_lots = []
            for lot in lots:
                t_idx, lot_shares, max_p, b_date, b_price = lot
                curr_p = p_t[t_idx]
                if curr_p > 0:
                    if curr_p > max_p:
                        max_p = curr_p
                    if curr_p <= max_p * (1.0 - sl_frac):
                        proceeds = lot_shares * curr_p
                        spare_cash += proceeds
                        running_holdings -= proceeds
                        lots_sold += 1
                        if collect_trades:
                            appr = (curr_p - b_price) / b_price * 100.0 if b_price > 0 else 0.0
                            ticker = str(TICKERS[t_idx])
                            transactions.append([d_str, ticker, "Sold", round(float(lot_shares), 6), round(float(curr_p), 2), round(float(proceeds), 2), round(float(running_holdings + spare_cash), 2)])
                            closed_lots.append([b_date, ticker, round(float(lot_shares), 6), round(float(b_price), 2), round(float(curr_p), 2), round(float(appr), 2), "SOLD", d_str])
                        continue
                remaining_lots.append([t_idx, lot_shares, max_p, b_date, b_price])
            lots = remaining_lots

        if contrib_mask[t]:
            p_now = prices[t]
            if cfg.direction == "benchmark":
                valid = ~np.isnan(p_now) & (p_now > 0)
                if np.any(valid):
                    alloc = cfg.amount / np.sum(valid)
                    if track_lots:
                        running_val = (sum(l[1] * p_t[l[0]] for l in lots) + spare_cash) if collect_trades else 0.0
                        for idx in np.where(valid)[0]:
                            sh = alloc / p_now[idx]
                            lots.append([int(idx), sh, p_now[idx], d_str, p_now[idx]])
                            if collect_trades:
                                trade_val = sh * p_now[idx]
                                running_val += trade_val
                                transactions.append([d_str, str(TICKERS[idx]), "Bought", round(float(sh), 6), round(float(p_now[idx]), 2), round(float(trade_val), 2), round(float(running_val), 2)])
                    else:
                        shares[valid] += alloc / p_now[valid]
                    contributed += cfg.amount
            else:
                p_prev = prices[t - lookback]
                valid = ~np.isnan(p_now) & ~np.isnan(p_prev) & (p_prev > 0) & (p_now > 0)
                if np.any(valid):
                    rets = np.where(valid, (p_now - p_prev) / p_prev, np.nan)
                    valid_idx = np.where(valid)[0]
                    if cfg.selection_model in ("low_vol_50", "low_vol_20") and t >= 20:
                        vol_win = min(t, 252)
                        sub_p = prices[t - vol_win : t + 1, :][:, valid_idx]
                        daily_r = np.diff(sub_p, axis=0) / np.where(sub_p[:-1] > 0, sub_p[:-1], np.nan)
                        vols = np.nanstd(daily_r, axis=0)
                        keep_frac = 0.50 if cfg.selection_model == "low_vol_50" else 0.20
                        keep_n = max(cfg.top_n, int(len(valid_idx) * keep_frac))
                        if keep_n < len(valid_idx):
                            calm_pos = np.argpartition(np.nan_to_num(vols, nan=np.inf), keep_n)[:keep_n]
                            valid_idx = valid_idx[calm_pos]
                    valid_rets = rets[valid_idx]
                    k = min(cfg.top_n, len(valid_idx))
                    if cfg.direction == "top":
                        chosen = valid_idx[np.argpartition(valid_rets, -k)[-k:]]
                    else:
                        chosen = valid_idx[np.argpartition(valid_rets, k)[:k]]
                    alloc = cfg.amount / k
                    if track_lots:
                        running_val = (sum(l[1] * p_t[l[0]] for l in lots) + spare_cash) if collect_trades else 0.0
                        for idx in chosen:
                            sh = alloc / p_now[idx]
                            lots.append([int(idx), sh, p_now[idx], d_str, p_now[idx]])
                            if collect_trades:
                                trade_val = sh * p_now[idx]
                                running_val += trade_val
                                transactions.append([d_str, str(TICKERS[idx]), "Bought", round(float(sh), 6), round(float(p_now[idx]), 2), round(float(trade_val), 2), round(float(running_val), 2)])
                    else:
                        shares[chosen] += alloc / p_now[chosen]
                    contributed += cfg.amount

        # Daily portfolio value (holdings + spare cash)
        if track_lots:
            holdings_val = sum(lot[1] * p_t[lot[0]] for lot in lots)
        else:
            holdings_val = np.dot(shares, p_t)
        equity_curve[t] = holdings_val + spare_cash
        contrib_curve[t] = contributed

    # Slice from first contribution day
    start_idx = lookback
    eq = equity_curve[start_idx:]
    co = contrib_curve[start_idx:]
    dt = dates[start_idx:]

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
    result = {
        "dates": [d.strftime("%Y-%m-%d") for d in dt[::step]],
        "equity": [round(float(x), 2) for x in eq[::step]],
        "contributed": [round(float(x), 2) for x in co[::step]],
        "metrics": {
            "final_value": round(final_val, 2),
            "holdings_value": round(final_val - spare_cash, 2),
            "spare_cash": round(spare_cash, 2),
            "lots_sold": lots_sold,
            "total_contributed": round(total_inv, 2),
            "profit": round(profit, 2),
            "roi_pct": round(roi_pct, 2),
            "cagr_pct": round(cagr_pct, 2),
            "max_drawdown_pct": round(max_dd, 2),
        }
    }
    if collect_trades:
        p_last = np.nan_to_num(prices[-1], nan=0.0)
        all_lots = list(closed_lots)
        for lot in lots:
            t_idx, lot_shares, _, b_date, b_price = lot
            curr_p = p_last[t_idx]
            appr = (curr_p - b_price) / b_price * 100.0 if b_price > 0 else 0.0
            all_lots.append([b_date, str(TICKERS[t_idx]), round(float(lot_shares), 6), round(float(b_price), 2), round(float(curr_p), 2), round(float(appr), 2), "OPEN", ""])
        result["transactions"] = transactions
        result["lots"] = all_lots
    return result

@app.post("/api/export/transactions")
def export_transactions(items: list[StrategyExportItem]):
    from collections import defaultdict
    cols_per_strat = ["Strategy", "Date", "Share", "Action", "No of Shares", "Price", "Total Price", "Portfolio Value"]
    strat_by_date = []
    all_dates = set()
    for item in items:
        if item.config.direction == "benchmark":
            continue
        res = _run_sim(item.config, collect_trades=True)
        by_date = defaultdict(list)
        for row in res["transactions"]:
            by_date[row[0]].append([item.name] + row)
            all_dates.add(row[0])
        strat_by_date.append(by_date)

    buf = io.StringIO()
    writer = csv.writer(buf)
    if strat_by_date:
        header = []
        for i in range(len(strat_by_date)):
            if i > 0:
                header.append("")
            header.extend(cols_per_strat)
        writer.writerow(header)

        empty_block = [""] * len(cols_per_strat)
        for d in sorted(all_dates):
            max_on_day = max(len(s[d]) for s in strat_by_date)
            for k in range(max_on_day):
                row_out = []
                for i, s in enumerate(strat_by_date):
                    if i > 0:
                        row_out.append("")
                    row_out.extend(s[d][k] if k < len(s[d]) else empty_block)
                writer.writerow(row_out)
    else:
        writer.writerow(cols_per_strat)

    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=transactions.csv"},
    )

@app.post("/api/export/lots")
def export_lots(items: list[StrategyExportItem]):
    rows = []
    for item in items:
        if item.config.direction == "benchmark":
            continue
        res = _run_sim(item.config, collect_trades=True)
        for row in res["lots"]:
            rows.append([item.name] + row)
    # Sort by % Appreciation descending (index 6 in [Strategy, Buy Date, Share, No of Shares, Share Bought Price, Current or Sold Price, % Appreciation, Status, Sold Date])
    rows.sort(key=lambda r: r[6], reverse=True)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Strategy", "Buy Date", "Share", "No of Shares", "Share Bought Price", "Current or Sold Price", "% Appreciation", "Status", "Sold Date"])
    writer.writerows(rows)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=lots_performance.csv"},
    )

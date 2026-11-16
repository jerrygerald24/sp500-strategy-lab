# S&P 500 DCA Momentum & Contrarian Investment Simulator

## 1. Project Overview & Objective
Build an interactive simulation and backtesting platform that analyzes the historical performance (last 10 years: 2016–2026) of dynamic Dollar-Cost Averaging (DCA) strategies on S&P 500 constituent stocks.

The platform simulates what happens when an investor systematically deploys capital based on prior-period winners (Momentum) vs. prior-period losers (Mean Reversion / Contrarian), benchmarked against a standard Buy-and-Hold S&P 500 (SPY/VOO) DCA strategy.

---

## 2. Key Architectural Learnings from Prior Project (Car Tracker)
1. **Isolated Workspace:** All code, databases, virtual environments, caches, and web assets belong strictly in `C:\Users\JPF-Mini\sp500_simulator` to keep other projects (like `car_tracker`) clean.
2. **Local Caching First:** Fetching 10 years of data for ~500 stocks over network is resource-intensive. Pull once, validate splits/dividends using **Adjusted Close**, and persist locally (SQLite / Parquet / DuckDB).
3. **Interactive, Zero-Lag UI:**
   - Standalone, self-contained HTML/JS dashboard using Chart.js or Plotly.
   - Pre-computed simulation matrices or fast client-side calculations for instantaneous slider/filter responses.
   - Deployable to GitHub Pages or local browser for mobile/desktop validation.

---

## 3. Functional Requirements & Strategy Rules

### 3.1 Investment Schedule & Cadence
- **Frequencies:**
  - **Daily:** Invest on every official NYSE/NASDAQ trading day (~252 days/year).
  - **Weekly:** Invest on the first trading day of every week (~52 weeks/year).
- **Default Contribution:** $100 per period (configurable via slider/input in the UI).

### 3.2 Strategy Baskets
At each contribution date $T$, rank all active S&P 500 stocks by their percentage return over the prior period ($T-1$):
1. **Top 1 Gainer (Pure Momentum):**
   - 100% of budget ($100) buys fractional shares of the #1 top-performing stock from the prior period.
2. **Top 5 Gainers (Diversified Momentum):**
   - Budget is split equally ($20 each) across the top 5 performing stocks.
3. **Top 1 Loser (Pure Contrarian / "Buy the Dip"):**
   - 100% of budget ($100) buys fractional shares of the #1 worst-performing stock from the prior period.
4. **Top 5 Losers (Diversified Contrarian):**
   - Budget is split equally ($20 each) across the 5 worst-performing stocks.
5. **Benchmark (Passive S&P 500 DCA):**
   - Same budget ($100) invested into SPY (or VOO) on the exact same schedule.

### 3.3 Financial Calculations & Metrics
- **Cumulative Contributions ($):** Total cash injected out-of-pocket over time.
- **Portfolio Equity Curve ($):** Daily market value of all accumulated share holdings.
- **Total ROI (%):** `(Final Portfolio Value - Total Contributed) / Total Contributed * 100`.
- **CAGR (%):** True Compound Annual Growth Rate.
- **Max Drawdown (%):** Maximum peak-to-trough decline during the 10-year holding period.
- **Sharpe / Sortino Ratio:** Risk-adjusted return measure vs. risk-free rate.
- **Stock Holdings Breakdown:** Most accumulated shares, top individual gainers, and top individual losers in the portfolio.

---

## 4. UI / UX Requirements
1. **Controls & Filters:**
   - Cadence toggle: `[ Daily ]` vs `[ Weekly ]`
   - Strategy selector: `[ Top 1 Gainer ]`, `[ Top 5 Gainers ]`, `[ Top 1 Loser ]`, `[ Top 5 Losers ]`
   - Contribution amount input: Default `$100`
   - Time horizon selector: `[ 3 Years ]`, `[ 5 Years ]`, `[ 10 Years ]`
2. **Visualizations:**
   - **Growth Comparison Chart:** Strategy Portfolio Value vs. Benchmark (SPY) vs. Cash Invested baseline.
   - **Drawdown Chart:** Underwater plot comparing risk profiles.
   - **Constituent Exposure:** Treemap or bar chart of top holdings accumulated over the 10 years.
3. **KPI Summary Cards:**
   - Ending Balance, Total Profit, ROI %, Annualized Return (CAGR), Max Drawdown.

---

## 5. Implementation Roadmap
1. **Phase 1: Data Pipeline (`fetch_sp500.py`)**
   - Fetch list of S&P 500 constituent tickers.
   - Download 10-year historical adjusted close prices from Yahoo Finance.
   - Save to `sp500_history.parquet` or `sp500_history.db`.
2. **Phase 2: Simulation Engine (`simulator.py`)**
   - Vectorized / fast backtester calculating daily share accumulations, portfolio values, and benchmark returns.
3. **Phase 3: Interactive Dashboard (`dashboard.html` / `app.py`)**
   - Build client-side or Streamlit interactive UX with instant reactivity.
4. **Phase 4: Validation & Testing**
   - Verify calculation correctness against known SPY historical returns.

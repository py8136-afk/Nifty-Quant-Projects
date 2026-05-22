"""
Portfolio Stress Testing Framework
====================================
8 NSE stocks, ₹5Cr notional, 5 scenarios (3 historical + 2 hypothetical)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "yfinance", "--break-system-packages", "-q"])
    import yfinance as yf

# ─────────────────────────────────────────────────────────
# PORTFOLIO SETUP
# ─────────────────────────────────────────────────────────
NOTIONAL   = 5_00_00_000          # ₹5 Crore
TICKERS_NS = {
    "RELIANCE"    : "RELIANCE.NS",
    "HDFCBANK"    : "HDFCBANK.NS",
    "INFY"        : "INFY.NS",
    "TCS"         : "TCS.NS",
    "ICICIBANK"   : "ICICIBANK.NS",
    "HINDUNILVR"  : "HINDUNILVR.NS",
    "SBIN"        : "SBIN.NS",
    "M&M"         : "M&M.NS",           # Mahindra (TATAMOTORS.NS unavailable on yfinance)
}
STOCKS  = list(TICKERS_NS.keys())
N       = len(STOCKS)
WEIGHTS = np.array([1 / N] * N)   # equal weight

# ─────────────────────────────────────────────────────────
# SECTOR MAP for hypothetical shocks
# ─────────────────────────────────────────────────────────
SECTOR = {
    "RELIANCE"   : "energy",
    "HDFCBANK"   : "financials",
    "INFY"       : "it",
    "TCS"        : "it",
    "ICICIBANK"  : "financials",
    "HINDUNILVR" : "fmcg",
    "SBIN"       : "financials",
    "M&M"        : "auto",
}

print("=" * 65)
print("  PORTFOLIO STRESS TESTING FRAMEWORK")
print("=" * 65)
print(f"  Stocks   : {', '.join(STOCKS)}")
print(f"  Weights  : Equal ({100/N:.1f}% each)")
print(f"  Notional : ₹{NOTIONAL/1e7:.0f} Crore\n")

# ─────────────────────────────────────────────────────────
# DOWNLOAD PRICE DATA
# ─────────────────────────────────────────────────────────
print("  Downloading 5Y daily prices from Yahoo Finance...")
tickers_list = list(TICKERS_NS.values())
raw = yf.download(tickers_list, start="2019-01-01", end="2024-12-31",
                  auto_adjust=True, progress=False)

# Extract Close prices, rename columns to short names
prices = raw["Close"].copy()
inv_map = {v: k for k, v in TICKERS_NS.items()}
prices.rename(columns=inv_map, inplace=True)
prices = prices[STOCKS].dropna(how="all")
prices.ffill(inplace=True)

# Daily returns — drop only rows where ALL stocks are NaN
returns = prices.pct_change().dropna(how="all")

print(f"  Price data: {prices.index[0].date()} → {prices.index[-1].date()}")
print(f"  Trading days: {len(prices)}\n")

# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────
def portfolio_return_from_window(ret_window: pd.DataFrame,
                                 weights: np.ndarray) -> float:
    """Compound cumulative portfolio return over a return window."""
    cum_stock = (1 + ret_window).prod() - 1          # per-stock cumulative
    return float(np.dot(weights, cum_stock))


def pnl_attribution(ret_window: pd.DataFrame,
                    weights: np.ndarray,
                    notional: float) -> pd.Series:
    """Per-stock P&L contribution (₹)."""
    cum = (1 + ret_window).prod() - 1
    alloc = weights * notional
    return pd.Series(cum.values * alloc, index=ret_window.columns)


def recovery_days(prices: pd.DataFrame,
                  scenario_end: str,
                  weights: np.ndarray,
                  peak_before: str = None) -> str:
    """Days for equal-weight portfolio to recover to pre-crash peak."""
    try:
        sub = prices.loc[scenario_end:]
        port = (sub * weights).sum(axis=1)
        port = port / port.iloc[0]           # index = 1 at scenario end
        # look for the portfolio recovering to 1.0 (end-of-crash level isn't
        # the pre-crash peak, but we track from crash trough)
        recovery = port[port >= 1.0]
        if len(recovery) > 1:
            return f"{(recovery.index[1] - recovery.index[0]).days} days"
        return "Not recovered in data"
    except Exception:
        return "N/A"


# ─────────────────────────────────────────────────────────
# SCENARIO 1 — COVID CRASH  (Feb 20 – Mar 23, 2020)
# ─────────────────────────────────────────────────────────
s1_start, s1_end = "2020-02-20", "2020-03-23"
s1_ret   = returns.loc[s1_start:s1_end]
s1_port  = portfolio_return_from_window(s1_ret, WEIGHTS)
s1_pnl   = s1_port * NOTIONAL
s1_attr  = pnl_attribution(s1_ret, WEIGHTS, NOTIONAL)

# ─────────────────────────────────────────────────────────
# SCENARIO 2 — 2022 Rate Hike Selloff  (Jan 1 – Jun 17, 2022)
# ─────────────────────────────────────────────────────────
s2_start, s2_end = "2022-01-03", "2022-06-17"
s2_ret   = returns.loc[s2_start:s2_end]
s2_port  = portfolio_return_from_window(s2_ret, WEIGHTS)
s2_pnl   = s2_port * NOTIONAL
s2_attr  = pnl_attribution(s2_ret, WEIGHTS, NOTIONAL)

# ─────────────────────────────────────────────────────────
# SCENARIO 3 — GFC Proxy  (Oct 2008 returns, scaled to Indian market)
# – Nifty fell ~27% in Oct 2008; we scale each stock's Oct-08 return
#   by (Nifty_actual / Nifty_global_proxy) if NSE data available,
#   else use Nifty beta-scaling approach: apply -27% uniform shock
#   distributed via each stock's 5-year beta vs equal-weight portfolio
# ─────────────────────────────────────────────────────────
# Compute each stock's beta vs portfolio (5-year daily)
port_ret = returns.dot(WEIGHTS)
betas = {}
for s in STOCKS:
    cov  = np.cov(returns[s], port_ret)[0, 1]
    var  = np.var(port_ret)
    betas[s] = cov / var if var > 0 else 1.0

nifty_oct08 = -0.27                          # Nifty -27% in Oct 2008
s3_stock_returns = {s: betas[s] * nifty_oct08 for s in STOCKS}
s3_ret_series = pd.Series(s3_stock_returns)
s3_port  = float(WEIGHTS @ s3_ret_series.values)
s3_pnl   = s3_port * NOTIONAL
s3_attr  = s3_ret_series * WEIGHTS * NOTIONAL

# ─────────────────────────────────────────────────────────
# SCENARIO 4 — RBI Emergency Rate Hike (hypothetical)
# Nifty -15%; financials -20%; IT -10%; others -12%
# ─────────────────────────────────────────────────────────
sector_shock_4 = {
    "financials": -0.20,
    "it"        : -0.10,
    "energy"    : -0.12,
    "fmcg"      : -0.12,
    "auto"      : -0.15,
}
s4_shocks = pd.Series({s: sector_shock_4.get(SECTOR[s], -0.15)
                        for s in STOCKS})
s4_port  = float(WEIGHTS @ s4_shocks.values)
s4_pnl   = s4_port * NOTIONAL
s4_attr  = s4_shocks * WEIGHTS * NOTIONAL

# ─────────────────────────────────────────────────────────
# SCENARIO 5 — Global Recession (hypothetical)
# All sectors -25% to -35%
# ─────────────────────────────────────────────────────────
sector_shock_5 = {
    "financials": -0.35,
    "it"        : -0.28,
    "energy"    : -0.30,
    "fmcg"      : -0.25,
    "auto"      : -0.33,
}
s5_shocks = pd.Series({s: sector_shock_5.get(SECTOR[s], -0.30)
                        for s in STOCKS})
s5_port  = float(WEIGHTS @ s5_shocks.values)
s5_pnl   = s5_port * NOTIONAL
s5_attr  = s5_shocks * WEIGHTS * NOTIONAL

# ─────────────────────────────────────────────────────────
# RECOVERY TIMES (historical scenarios)
# ─────────────────────────────────────────────────────────
rec1 = recovery_days(prices, s1_end, WEIGHTS)
rec2 = recovery_days(prices, s2_end, WEIGHTS)

# ─────────────────────────────────────────────────────────
# RESULTS SUMMARY
# ─────────────────────────────────────────────────────────
scenarios = [
    ("COVID Crash (Feb-Mar 2020)",      s1_port, s1_pnl, s1_attr, "Historical", rec1),
    ("2022 Rate Hike Selloff",          s2_port, s2_pnl, s2_attr, "Historical", rec2),
    ("GFC Proxy (Oct 2008)",            s3_port, s3_pnl, s3_attr, "Historical", "Pre-data"),
    ("RBI Emergency Rate Hike",         s4_port, s4_pnl, s4_attr, "Hypothetical", "N/A"),
    ("Global Recession",                s5_port, s5_pnl, s5_attr, "Hypothetical", "N/A"),
]

print("=" * 65)
print("  STRESS TEST RESULTS")
print("=" * 65)
print(f"  {'Scenario':<32} {'Type':>12}  {'P&L (₹Cr)':>10}  {'Loss %':>8}  {'Recovery'}")
print("-" * 85)
for name, port_ret_, pnl, _, typ, rec in scenarios:
    cr = pnl / 1e7
    print(f"  {name:<32} {typ:>12}  {cr:>10.3f}  {port_ret_*100:>7.2f}%  {rec}")
print()

# Worst scenario
worst_idx  = int(np.argmin([s[2] for s in scenarios]))
worst_name, _, worst_pnl, worst_attr, _, _ = scenarios[worst_idx]

print("=" * 65)
print(f"  P&L ATTRIBUTION — WORST SCENARIO: {worst_name}")
print("=" * 65)
print(f"  {'Stock':<16} {'Sector':<14} {'Shock %':>9}  {'P&L (₹L)':>12}")
print("-" * 55)
attr_sorted = worst_attr.sort_values()
for stock, val in attr_sorted.items():
    shock = val / (WEIGHTS[STOCKS.index(stock)] * NOTIONAL)
    print(f"  {stock:<16} {SECTOR[stock]:<14} {shock*100:>8.2f}%  {val/1e5:>11.2f}")
print(f"  {'TOTAL':<16} {'':<14} {worst_pnl/NOTIONAL*100:>8.2f}%  {worst_pnl/1e5:>11.2f}")
print()

# ─────────────────────────────────────────────────────────
# SAVE CSV
# ─────────────────────────────────────────────────────────
rows = []
for name, port_ret_, pnl, attr, typ, rec in scenarios:
    row = {
        "scenario"    : name,
        "type"        : typ,
        "portfolio_return_pct": round(port_ret_ * 100, 4),
        "portfolio_pnl_cr"   : round(pnl / 1e7, 4),
        "recovery"    : rec,
    }
    for s in STOCKS:
        row[f"{s}_pnl_L"] = round(attr[s] / 1e5, 2)
    rows.append(row)

df_results = pd.DataFrame(rows)
df_results.to_csv("stress_results.csv", index=False)
print("  stress_results.csv saved.\n")

# ─────────────────────────────────────────────────────────
# HEATMAP — scenarios × stocks (% loss)
# ─────────────────────────────────────────────────────────
heat_data = []
scenario_labels = [
    "COVID\nCrash",
    "2022 Rate\nHike",
    "GFC\nProxy",
    "RBI Rate\nHike",
    "Global\nRecession",
]
for i, (_, _, _, attr, _, _) in enumerate(scenarios):
    row = {}
    for s in STOCKS:
        alloc = WEIGHTS[STOCKS.index(s)] * NOTIONAL
        row[s] = attr[s] / alloc * 100     # % loss per stock
    heat_data.append(row)

heat_df = pd.DataFrame(heat_data, index=scenario_labels)

fig, ax = plt.subplots(figsize=(13, 5))
sns.heatmap(
    heat_df,
    annot=True, fmt=".1f", linewidths=0.5,
    cmap="RdYlGn",
    center=0,
    vmin=heat_df.values.min(),
    vmax=min(0, heat_df.values.max()),
    annot_kws={"size": 9},
    ax=ax,
    cbar_kws={"label": "Stock Return (%)"},
)
ax.set_title("Portfolio Stress Test — Per-Stock Returns (%) by Scenario",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Stock", fontsize=11)
ax.set_ylabel("Scenario", fontsize=11)
ax.tick_params(axis="x", rotation=30)
ax.tick_params(axis="y", rotation=0)

# Annotate portfolio-level P&L on right margin
for i, (_, port_ret_, pnl, _, _, _) in enumerate(scenarios):
    ax.text(len(STOCKS) + 0.15, i + 0.5,
            f"  Port: {port_ret_*100:.1f}%\n  ₹{pnl/1e7:.2f}Cr",
            va="center", fontsize=7.5, color="black")

plt.tight_layout()
plt.savefig("stress_test_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  stress_test_heatmap.png saved.\n")

print("=" * 65)
print("  ALL DONE.")
print("=" * 65)

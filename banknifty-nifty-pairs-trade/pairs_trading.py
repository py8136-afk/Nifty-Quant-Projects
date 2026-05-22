"""
BankNifty-Nifty Cointegration Pairs Trading System
Data: Jan 2020 – May 2025 | Daily closing prices
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sqlite3
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
OUTDIR       = os.path.dirname(os.path.abspath(__file__))
START        = "2020-01-01"
END          = "2025-05-22"
ROLL_WIN     = 60          # rolling OLS / z-score window
ENTRY_Z      = 2.0
EXIT_Z       = 0.5
PNL_PER_UNIT = 1.0         # ₹1 per unit of spread
DB_PATH      = os.path.join(OUTDIR, "trades.db")
CSV_PATH     = os.path.join(OUTDIR, "pairs_results.csv")
PLOT_Z       = os.path.join(OUTDIR, "zscore_signals.png")
PLOT_EQ      = os.path.join(OUTDIR, "equity_curve.png")

# ─────────────────────────────────────────────
# STEP 0 — DATA DOWNLOAD
# ─────────────────────────────────────────────
print("=" * 65)
print("  BankNifty–Nifty Cointegration Pairs Trading System")
print("=" * 65)
print(f"\n[DATA] Downloading ^NSEBANK and ^NSEI  ({START} → {END})…")

raw = yf.download(["^NSEBANK", "^NSEI"], start=START, end=END,
                  auto_adjust=True, progress=False)["Close"]
raw.columns = ["BankNifty", "Nifty"]
raw.dropna(inplace=True)
raw.index = pd.to_datetime(raw.index)

print(f"       Rows after dropping NaN: {len(raw)}")
print(f"       Date range : {raw.index[0].date()} → {raw.index[-1].date()}")
print(raw.tail(3).to_string())

BN = raw["BankNifty"]
NF = raw["Nifty"]

# ─────────────────────────────────────────────
# STEP 1 — COINTEGRATION TEST (Engle-Granger)
# ─────────────────────────────────────────────
print("\n" + "─" * 65)
print("STEP 1 — Engle-Granger Cointegration Test")
print("─" * 65)

eg_stat, eg_pval, eg_crit = coint(BN, NF)
print(f"  EG Statistic : {eg_stat:.4f}")
print(f"  p-value      : {eg_pval:.6f}")
print(f"  Critical vals: 1%={eg_crit[0]:.4f}  5%={eg_crit[1]:.4f}  10%={eg_crit[2]:.4f}")
cointegrated = eg_pval < 0.05
print(f"  Cointegrated at 5%: {'YES ✓' if cointegrated else 'NO ✗'}")

# Static OLS for spread (used only for ADF check)
static_model = sm.OLS(BN, sm.add_constant(NF)).fit()
static_hedge  = static_model.params["Nifty"]
static_spread = BN - static_hedge * NF

print("\n  ADF Test on static spread:")
adf_stat, adf_pval, adf_lags, _, adf_crit, _ = adfuller(static_spread.dropna(), autolag="AIC")
print(f"  ADF Statistic : {adf_stat:.4f}")
print(f"  p-value       : {adf_pval:.6f}")
print(f"  Lags used     : {adf_lags}")
print(f"  Critical vals : 1%={adf_crit['1%']:.4f}  5%={adf_crit['5%']:.4f}  10%={adf_crit['10%']:.4f}")
stationary = adf_pval < 0.05
print(f"  Spread stationary at 5%: {'YES ✓' if stationary else 'NO ✗'}")

# ─────────────────────────────────────────────
# STEP 2 — ROLLING HEDGE RATIO & Z-SCORE
# ─────────────────────────────────────────────
print("\n" + "─" * 65)
print("STEP 2 — Rolling Hedge Ratio (60-day OLS) & Z-Score")
print("─" * 65)

df = raw.copy()

# Rolling OLS: BankNifty ~ Nifty
endog  = df["BankNifty"]
exog   = sm.add_constant(df["Nifty"])
rol    = RollingOLS(endog, exog, window=ROLL_WIN).fit()

df["hedge_ratio"] = rol.params["Nifty"]
df["spread"]      = df["BankNifty"] - df["hedge_ratio"] * df["Nifty"]

# Rolling z-score
df["spread_mean"] = df["spread"].rolling(ROLL_WIN).mean()
df["spread_std"]  = df["spread"].rolling(ROLL_WIN).std()
df["zscore"]      = (df["spread"] - df["spread_mean"]) / df["spread_std"]

df.dropna(inplace=True)
print(f"  Rows after warm-up period: {len(df)}")
print(f"  Hedge ratio — mean={df['hedge_ratio'].mean():.4f}  "
      f"min={df['hedge_ratio'].min():.4f}  max={df['hedge_ratio'].max():.4f}")
print(f"  Z-score     — mean={df['zscore'].mean():.4f}  "
      f"std={df['zscore'].std():.4f}  range=[{df['zscore'].min():.2f}, {df['zscore'].max():.2f}]")

# ─────────────────────────────────────────────
# STEP 3 & 4 — SIGNAL GENERATION & BACKTEST
# ─────────────────────────────────────────────
print("\n" + "─" * 65)
print("STEP 3 & 4 — Signal Generation & Backtest")
print("─" * 65)

trades        = []       # completed trade records
position      = 0        # 0=flat, 1=long spread, -1=short spread
entry_date    = None
entry_z       = None
entry_spread  = None
direction_str = None

entry_signals  = []      # (date, z) for plotting
exit_signals   = []      # (date, z) for plotting

df_vals = df[["zscore", "spread"]].copy()

for date, row in df_vals.iterrows():
    z = row["zscore"]
    sp = row["spread"]

    if position == 0:
        if z < -ENTRY_Z:
            position      = 1
            entry_date    = date
            entry_z       = z
            entry_spread  = sp
            direction_str = "LONG"
            entry_signals.append((date, z))
        elif z > ENTRY_Z:
            position      = -1
            entry_date    = date
            entry_z       = z
            entry_spread  = sp
            direction_str = "SHORT"
            entry_signals.append((date, z))
    else:
        crossed_exit = (position == 1  and z >= -EXIT_Z) or \
                       (position == -1 and z <=  EXIT_Z)
        if crossed_exit:
            pnl = (sp - entry_spread) * position * PNL_PER_UNIT
            trades.append({
                "entry_date"  : entry_date,
                "exit_date"   : date,
                "direction"   : direction_str,
                "entry_zscore": entry_z,
                "exit_zscore" : z,
                "entry_spread": entry_spread,
                "exit_spread" : sp,
                "pnl"         : pnl,
                "exit_reason" : "z-score mean reversion",
            })
            exit_signals.append((date, z))
            position = 0

# Close any open position at the last bar
if position != 0:
    last_date   = df_vals.index[-1]
    last_z      = df_vals["zscore"].iloc[-1]
    last_spread = df_vals["spread"].iloc[-1]
    pnl = (last_spread - entry_spread) * position * PNL_PER_UNIT
    trades.append({
        "entry_date"  : entry_date,
        "exit_date"   : last_date,
        "direction"   : direction_str,
        "entry_zscore": entry_z,
        "exit_zscore" : last_z,
        "entry_spread": entry_spread,
        "exit_spread" : last_spread,
        "pnl"         : pnl,
        "exit_reason" : "end of data",
    })

trades_df = pd.DataFrame(trades)
trades_df.insert(0, "id", range(1, len(trades_df) + 1))

# ── Equity curve ──────────────────────────────
equity   = pd.Series(0.0, index=df.index)
for _, t in trades_df.iterrows():
    eq_dates = df.index[(df.index >= t["entry_date"]) & (df.index <= t["exit_date"])]
    if len(eq_dates) >= 2:
        daily_pl = t["pnl"] / (len(eq_dates) - 1)
        for d in eq_dates[1:]:
            equity[d] += daily_pl

equity_curve = equity.cumsum()

# ── Performance metrics ────────────────────────
total_return   = equity_curve.iloc[-1]
n_days         = len(equity_curve)
ann_factor     = 252 / n_days
ann_return     = total_return * ann_factor

daily_ret = equity.values
sr_denom   = daily_ret.std()
sharpe     = (daily_ret.mean() / sr_denom * np.sqrt(252)) if sr_denom > 0 else 0.0

roll_max   = equity_curve.cummax()
drawdowns  = equity_curve - roll_max
max_dd     = drawdowns.min()

n_trades   = len(trades_df)
win_trades = (trades_df["pnl"] > 0).sum()
win_rate   = win_trades / n_trades * 100 if n_trades > 0 else 0.0

print(f"\n  {'Metric':<30} {'Value':>15}")
print(f"  {'─'*45}")
print(f"  {'Total Trades':<30} {n_trades:>15}")
print(f"  {'Winning Trades':<30} {win_trades:>15}")
print(f"  {'Win Rate':<30} {win_rate:>14.2f}%")
print(f"  {'Total P&L (₹)':<30} {total_return:>15.2f}")
print(f"  {'Annualised Return (₹)':<30} {ann_return:>15.2f}")
print(f"  {'Sharpe Ratio':<30} {sharpe:>15.4f}")
print(f"  {'Max Drawdown (₹)':<30} {max_dd:>15.2f}")

# ─────────────────────────────────────────────
# STEP 5 — SQLITE TRADE LOG
# ─────────────────────────────────────────────
print("\n" + "─" * 65)
print("STEP 5 — SQLite Trade Log")
print("─" * 65)

if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

conn = sqlite3.connect(DB_PATH)
cur  = conn.cursor()

cur.execute("""
    CREATE TABLE trades (
        id           INTEGER PRIMARY KEY,
        entry_date   TEXT,
        exit_date    TEXT,
        direction    TEXT,
        entry_zscore REAL,
        exit_zscore  REAL,
        entry_spread REAL,
        exit_spread  REAL,
        pnl          REAL,
        exit_reason  TEXT
    )
""")

for _, row in trades_df.iterrows():
    cur.execute("""
        INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (
        int(row["id"]),
        str(row["entry_date"].date()),
        str(row["exit_date"].date()),
        row["direction"],
        float(row["entry_zscore"]),
        float(row["exit_zscore"]),
        float(row["entry_spread"]),
        float(row["exit_spread"]),
        float(row["pnl"]),
        row["exit_reason"],
    ))

conn.commit()
print(f"  Inserted {n_trades} trades into {DB_PATH}")

# ── Query 1: by direction ─────────────────────
print("\n  Query 1 — P&L by Direction:")
q1 = cur.execute("""
    SELECT direction,
           COUNT(*)   AS num_trades,
           ROUND(AVG(pnl),2) AS avg_pnl,
           ROUND(SUM(pnl),2) AS total_pnl
    FROM trades
    GROUP BY direction
""").fetchall()
print(f"  {'Direction':<12} {'Count':>8} {'Avg PnL':>12} {'Total PnL':>12}")
print(f"  {'─'*46}")
for r in q1:
    print(f"  {r[0]:<12} {r[1]:>8} {r[2]:>12.2f} {r[3]:>12.2f}")

# ── Query 2: annual P&L ───────────────────────
print("\n  Query 2 — Annual P&L:")
q2 = cur.execute("""
    SELECT strftime('%Y', entry_date) AS year,
           ROUND(SUM(pnl),2)         AS annual_pnl
    FROM trades
    GROUP BY year
    ORDER BY year
""").fetchall()
print(f"  {'Year':<8} {'Annual PnL':>14}")
print(f"  {'─'*24}")
for r in q2:
    print(f"  {r[0]:<8} {r[1]:>14.2f}")

# ── Query 3: best & worst trade ───────────────
print("\n  Query 3 — Best & Worst Trade:")
q3 = cur.execute("""
    SELECT * FROM trades
    WHERE pnl = (SELECT MAX(pnl) FROM trades)
       OR pnl = (SELECT MIN(pnl) FROM trades)
    ORDER BY pnl DESC
""").fetchall()
cols = ["id","entry_date","exit_date","direction","entry_z","exit_z",
        "entry_spread","exit_spread","pnl","exit_reason"]
print(f"  {' | '.join(f'{c}' for c in cols)}")
print(f"  {'─'*100}")
for r in q3:
    print(f"  {' | '.join(str(v) for v in r)}")

conn.close()

# ─────────────────────────────────────────────
# SAVE pairs_results.csv
# ─────────────────────────────────────────────
df_out = df[["BankNifty","Nifty","hedge_ratio","spread","zscore"]].copy()
df_out.to_csv(CSV_PATH)
print(f"\n[CSV] Saved → {CSV_PATH}")

# ─────────────────────────────────────────────
# PLOT 1 — Z-Score with Entry/Exit Signals
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df.index, df["zscore"], color="#1f77b4", lw=0.9, label="Z-Score")
ax.axhline( ENTRY_Z, color="red",   lw=1.0, ls="--", label=f"+{ENTRY_Z} (Short entry)")
ax.axhline(-ENTRY_Z, color="green", lw=1.0, ls="--", label=f"-{ENTRY_Z} (Long entry)")
ax.axhline( EXIT_Z,  color="orange",lw=0.8, ls=":",  label=f"±{EXIT_Z} (Exit band)")
ax.axhline(-EXIT_Z,  color="orange",lw=0.8, ls=":")
ax.axhline(0,        color="black", lw=0.6, ls="-",  alpha=0.4)

if entry_signals:
    ed, ez = zip(*entry_signals)
    long_e  = [(d,z) for d,z in zip(ed,ez) if z < 0]
    short_e = [(d,z) for d,z in zip(ed,ez) if z > 0]
    if long_e:
        ld,lz = zip(*long_e)
        ax.scatter(ld, lz, marker="^", color="green", s=60, zorder=5, label="Long entry")
    if short_e:
        sd,sz = zip(*short_e)
        ax.scatter(sd, sz, marker="v", color="red",   s=60, zorder=5, label="Short entry")

if exit_signals:
    xd, xz = zip(*exit_signals)
    ax.scatter(xd, xz, marker="x", color="black", s=50, zorder=5, label="Exit")

ax.set_title("BankNifty–Nifty Spread Z-Score with Entry/Exit Signals", fontsize=14, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Z-Score")
ax.legend(loc="upper right", fontsize=8, ncol=2)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=30)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_Z, dpi=150)
plt.close()
print(f"[PLOT] Z-score chart saved → {PLOT_Z}")

# ─────────────────────────────────────────────
# PLOT 2 — Equity Curve
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(equity_curve.index, equity_curve.values, color="#2ca02c", lw=1.2, label="Equity Curve")
ax.fill_between(equity_curve.index, equity_curve.values,
                where=(equity_curve.values >= 0), alpha=0.15, color="green")
ax.fill_between(equity_curve.index, equity_curve.values,
                where=(equity_curve.values < 0),  alpha=0.15, color="red")
ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.6)

# Annotate final value
ax.annotate(f"Final: ₹{equity_curve.iloc[-1]:.0f}",
            xy=(equity_curve.index[-1], equity_curve.iloc[-1]),
            xytext=(-80, -20), textcoords="offset points",
            fontsize=10, color="#2ca02c",
            arrowprops=dict(arrowstyle="->", color="#2ca02c"))

# Drawdown shading
roll_max2 = equity_curve.cummax()
dd2       = equity_curve - roll_max2
max_dd_date = dd2.idxmin()
ax2 = ax.twinx()
ax2.fill_between(dd2.index, dd2.values, 0, alpha=0.2, color="red", label="Drawdown")
ax2.set_ylabel("Drawdown (₹)", color="red")
ax2.tick_params(axis="y", labelcolor="red")
ax2.set_ylim(dd2.min() * 4, 0)

textbox = (f"Trades: {n_trades}  |  Win Rate: {win_rate:.1f}%\n"
           f"Total P&L: ₹{total_return:.0f}  |  Sharpe: {sharpe:.2f}\n"
           f"Max DD: ₹{max_dd:.0f}")
ax.text(0.02, 0.97, textbox, transform=ax.transAxes, fontsize=9,
        verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

ax.set_title("BankNifty–Nifty Pairs Trading — Equity Curve", fontsize=14, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative P&L (₹)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=30)
ax.grid(alpha=0.3)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(PLOT_EQ, dpi=150)
plt.close()
print(f"[PLOT] Equity curve saved → {PLOT_EQ}")

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  FINAL PERFORMANCE SUMMARY")
print("=" * 65)
print(f"  Period              : {raw.index[0].date()} → {raw.index[-1].date()}")
print(f"  Cointegrated (5%)   : {'YES' if cointegrated else 'NO'}")
print(f"  Spread Stationary   : {'YES' if stationary else 'NO'}")
print(f"  Total Trades        : {n_trades}")
print(f"  Winning Trades      : {win_trades}  ({win_rate:.1f}%)")
print(f"  Total P&L           : ₹{total_return:.2f}")
print(f"  Annualised Return   : ₹{ann_return:.2f}")
print(f"  Sharpe Ratio        : {sharpe:.4f}")
print(f"  Max Drawdown        : ₹{max_dd:.2f}")
print(f"  Avg P&L per Trade   : ₹{trades_df['pnl'].mean():.2f}")
print(f"  Best Trade          : ₹{trades_df['pnl'].max():.2f}")
print(f"  Worst Trade         : ₹{trades_df['pnl'].min():.2f}")
print("=" * 65)
print("\nAll outputs saved:")
print(f"  CSV  → {CSV_PATH}")
print(f"  DB   → {DB_PATH}")
print(f"  Plot → {PLOT_Z}")
print(f"  Plot → {PLOT_EQ}")

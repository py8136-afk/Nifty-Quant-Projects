"""
Portfolio Risk Decomposition via PCA — Nifty 50 (10 stocks)
=============================================================
Covariance → Marginal/Component VaR → Diversification Ratio → PCA
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
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
# PARAMETERS
# ─────────────────────────────────────────────────────────
NOTIONAL   = 1_00_00_000          # ₹1 Crore
TICKERS    = [
    "RELIANCE.NS", "HDFCBANK.NS", "INFY.NS",    "TCS.NS",
    "ICICIBANK.NS","HINDUNILVR.NS","SBIN.NS",    "WIPRO.NS",
    "AXISBANK.NS", "BAJFINANCE.NS"
]
NAMES = [t.replace(".NS", "") for t in TICKERS]
N          = len(TICKERS)
WEIGHTS    = np.full(N, 1.0 / N)
TRADING_DAYS = 252
Z_99       = norm.ppf(0.99)        # 2.3263

def hdr(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)

# ─────────────────────────────────────────────────────────
# DATA DOWNLOAD
# ─────────────────────────────────────────────────────────
hdr("DATA DOWNLOAD")
print("  Fetching Jan 2020 – May 2025 daily prices …")

raw = yf.download(TICKERS, start="2020-01-01", end="2025-05-22",
                  auto_adjust=True, progress=False)
prices = raw["Close"].copy()
prices.columns = [c.replace(".NS", "") for c in prices.columns]
prices = prices[NAMES].ffill().dropna()

log_ret = np.log(prices / prices.shift(1)).dropna()

print(f"  Stocks       : {', '.join(NAMES)}")
print(f"  Date range   : {log_ret.index[0].date()} → {log_ret.index[-1].date()}")
print(f"  Trading days : {len(log_ret)}")
print(f"  Notional     : ₹{NOTIONAL/1e7:.0f} Crore  (equal weights {100/N:.1f}% each)")


# ============================================================
# STEP 1 — COVARIANCE MATRIX & PORTFOLIO VOL
# ============================================================
hdr("STEP 1 — COVARIANCE MATRIX & PORTFOLIO VOLATILITY")

cov_ann  = log_ret.cov() * TRADING_DAYS          # annualised
corr_mat = log_ret.corr()

port_var  = WEIGHTS @ cov_ann.values @ WEIGHTS
port_vol  = np.sqrt(port_var)
port_vol_daily = port_vol / np.sqrt(TRADING_DAYS)

indiv_vols = np.sqrt(np.diag(cov_ann.values))    # annualised
wtd_vol_sum = WEIGHTS @ indiv_vols

print(f"\n  Annualised Portfolio Volatility : {port_vol*100:.2f}%")
print(f"  Portfolio Variance              : {port_var*100:.4f}% (ann.)")
print(f"  Daily Portfolio Vol             : {port_vol_daily*100:.2f}%")
print(f"\n  Individual annualised vols:")
for i, nm in enumerate(NAMES):
    print(f"    {nm:<14}: {indiv_vols[i]*100:.2f}%")

print(f"\n  INSIGHT: Equal-weight portfolio vol {port_vol*100:.2f}% vs "
      f"simple avg of individual vols {wtd_vol_sum*100:.2f}% — "
      f"diversification saves {(wtd_vol_sum - port_vol)*100:.2f} vol points.")

# Save annualised covariance table
cov_df = pd.DataFrame(cov_ann.values, index=NAMES, columns=NAMES)


# ============================================================
# STEP 2 — MARGINAL VaR
# ============================================================
hdr("STEP 2 — MARGINAL VaR  (99% confidence, z = 2.3263)")

# MVaR_i = (Σw)_i / σ_p × z  ×  w_i   — per ₹1 of stock i
sigma_w   = cov_ann.values @ WEIGHTS          # (Σw) vector
beta_port = sigma_w / port_vol                # ∂σ_p/∂w_i  (correlation to portfolio)
mvar      = beta_port * Z_99                  # marginal VaR per unit weight

print(f"\n  {'Stock':<14}  {'Ann.Vol%':>9}  {'Beta_port':>10}  {'MargVaR(₹/₹)':>14}")
print(f"  {'-'*55}")
for i, nm in enumerate(NAMES):
    print(f"  {nm:<14}  {indiv_vols[i]*100:>9.2f}  {beta_port[i]:>10.4f}  {mvar[i]:>14.6f}")

print(f"\n  INSIGHT: {NAMES[np.argmax(mvar)]} has the highest marginal VaR "
      f"({mvar.max():.4f}) — adding more of it increases portfolio risk most.")


# ============================================================
# STEP 3 — COMPONENT VaR
# ============================================================
hdr("STEP 3 — COMPONENT VaR  (risk attribution)")

# Portfolio VaR (₹)
port_VaR_pct = port_vol * Z_99               # as fraction
port_VaR_rs  = port_VaR_pct * NOTIONAL       # in ₹

# Component VaR_i = w_i × MVaR_i × Notional
cvar_pct   = WEIGHTS * mvar                  # fraction of portfolio
cvar_rs    = cvar_pct * NOTIONAL             # ₹ contribution
cvar_share = cvar_pct / port_VaR_pct * 100  # % of total VaR

# Verify identity: sum(cvar) = portfolio VaR
identity_check = cvar_pct.sum()

print(f"\n  Portfolio 99% 1-Day VaR : ₹{port_VaR_rs/1e5:.4f}L  ({port_VaR_pct*100:.3f}% of notional)")
print(f"  Sum of Component VaRs   : ₹{(cvar_rs.sum())/1e5:.4f}L  — identity holds: "
      f"{'✓' if abs(identity_check - port_VaR_pct) < 1e-10 else '✗'}")
print(f"\n  {'Stock':<14}  {'CompVaR(₹L)':>12}  {'% of Total':>11}  {'Rank':>5}")
print(f"  {'-'*50}")
ranks = np.argsort(-cvar_share)
for rank, i in enumerate(ranks):
    nm = NAMES[i]
    print(f"  {nm:<14}  {cvar_rs[i]/1e5:>12.4f}  {cvar_share[i]:>10.2f}%  {rank+1:>5}")
print(f"  {'TOTAL':<14}  {cvar_rs.sum()/1e5:>12.4f}  {cvar_share.sum():>10.2f}%")

print(f"\n  INSIGHT: Top 3 risk contributors account for "
      f"{cvar_share[ranks[:3]].sum():.1f}% of portfolio VaR despite "
      f"being only {3/N*100:.0f}% of holdings.")


# ============================================================
# STEP 4 — DIVERSIFICATION RATIO
# ============================================================
hdr("STEP 4 — DIVERSIFICATION RATIO")

DR = wtd_vol_sum / port_vol
risk_eliminated_abs = (wtd_vol_sum - port_vol) * 100
risk_eliminated_pct = (1 - port_vol / wtd_vol_sum) * 100

# Variance if all stocks had correlation 1 (no diversification)
port_var_no_div = (WEIGHTS @ indiv_vols)**2
# Variance decomposition: diversifiable portion
systematic_var = port_var
total_undiversified_var = port_var_no_div
diversified_away = (1 - systematic_var / total_undiversified_var) * 100

print(f"\n  Weighted avg individual vol : {wtd_vol_sum*100:.2f}% p.a.")
print(f"  Portfolio vol (equal-wt)    : {port_vol*100:.2f}% p.a.")
print(f"  Diversification Ratio       : {DR:.4f}")
print(f"  Risk eliminated (abs)       : {risk_eliminated_abs:.2f} vol points p.a.")
print(f"  Risk eliminated (%)         : {risk_eliminated_pct:.2f}% of standalone vol")
print(f"  Portfolio var vs undiversified: {diversified_away:.2f}% variance eliminated")
print(f"\n  INSIGHT: DR = {DR:.2f} means the portfolio benefits from "
      f"{risk_eliminated_pct:.1f}% risk reduction vs holding each stock independently. "
      f"Correlation structure saves ₹{(wtd_vol_sum - port_vol)*NOTIONAL*Z_99/1e5:.2f}L in VaR.")


# ============================================================
# STEP 5 — PCA ON RETURN CORRELATION MATRIX
# ============================================================
hdr("STEP 5 — PCA ON 10×10 CORRELATION MATRIX")

# Standardise returns → correlation-based PCA
scaler    = StandardScaler()
ret_std   = scaler.fit_transform(log_ret.values)    # zero-mean, unit-var

pca       = PCA(n_components=N)
pca.fit(ret_std)

loadings     = pd.DataFrame(
    pca.components_.T,
    index   = NAMES,
    columns = [f"PC{i+1}" for i in range(N)]
)
expl_var     = pca.explained_variance_ratio_ * 100
cum_expl_var = np.cumsum(expl_var)
eigenvalues  = pca.explained_variance_

print(f"\n  {'PC':<6}  {'Eigenvalue':>12}  {'Var Explained':>14}  {'Cumulative':>12}")
print(f"  {'-'*50}")
for i in range(N):
    marker = " ◄ market factor" if i == 0 else (" ◄ sector rotation" if i == 1 else "")
    print(f"  PC{i+1:<4}  {eigenvalues[i]:>12.4f}  {expl_var[i]:>13.2f}%  {cum_expl_var[i]:>11.2f}%{marker}")

n_pcs_90 = np.searchsorted(cum_expl_var, 90) + 1
print(f"\n  PCs to explain 90% of variance: {n_pcs_90}")
print(f"  PC1 alone explains             : {expl_var[0]:.2f}%")
print(f"  PC1 + PC2                      : {cum_expl_var[1]:.2f}%")

# Factor loading interpretation
print(f"\n  PC1 FACTOR LOADINGS (market factor — all same sign expected):")
print(f"  {'Stock':<14}  {'PC1 loading':>12}  {'PC2 loading':>12}")
print(f"  {'-'*42}")
for nm in NAMES:
    print(f"  {nm:<14}  {loadings.loc[nm,'PC1']:>12.4f}  {loadings.loc[nm,'PC2']:>12.4f}")

# Economic interpretation
pc1_sign  = "positive" if loadings["PC1"].mean() > 0 else "negative"
top_pc1   = loadings["PC1"].abs().sort_values(ascending=False).index[:3].tolist()
top_pc2   = loadings["PC2"].abs().sort_values(ascending=False).index[:3].tolist()
print(f"\n  INSIGHT — PC1 ({expl_var[0]:.1f}% var): All loadings {pc1_sign} → pure market/beta factor.")
print(f"    Highest PC1 exposure: {', '.join(top_pc1)}")

# PC2 split: check if financials vs IT divide emerges
financials = ["HDFCBANK","ICICIBANK","SBIN","AXISBANK","BAJFINANCE"]
it_stocks  = ["INFY","TCS","WIPRO"]
pc2_fin = loadings.loc[[f for f in financials if f in NAMES], "PC2"].mean()
pc2_it  = loadings.loc[[f for f in it_stocks  if f in NAMES], "PC2"].mean()
pc2_story = ("Financials vs IT/Tech split" if abs(pc2_fin - pc2_it) > 0.2
             else "Growth vs Defensive split")
print(f"  INSIGHT — PC2 ({expl_var[1]:.1f}% var): {pc2_story}.")
print(f"    Financials avg PC2: {pc2_fin:.3f}  |  IT avg PC2: {pc2_it:.3f}")
print(f"    Highest PC2 exposure: {', '.join(top_pc2)}")


# ============================================================
# STEP 6 — PC-BASED RISK DECOMPOSITION
# ============================================================
hdr("STEP 6 — PC-BASED PORTFOLIO VARIANCE DECOMPOSITION")

# Portfolio weights projected onto PCs
# PC_score_portfolio = w' × loadings (each column)
# Var_from_PCk = (w' × loadings[:,k])^2 × eigenvalue_k / N
# (loadings are normalised eigenvectors of correlation matrix)

# Transform portfolio into PC space
# portfolio PC exposure = loadings.T @ (w × vol_normalised)
# Use standardised returns: each stock scaled by its own std
indiv_std_daily = log_ret.std().values
w_normalised    = WEIGHTS * indiv_std_daily           # w scaled by individual vols

pc_port_exposure = loadings.values.T @ w_normalised   # shape (N,)

# Variance contribution from each PC
# Total portfolio variance (in standardised units) = sum_k λ_k * (w'v_k)^2
# Convert back: multiply by scaler factor
pc_var_contrib   = eigenvalues * pc_port_exposure**2
total_pc_var     = pc_var_contrib.sum()
pc_var_share     = pc_var_contrib / total_pc_var * 100

# Undiversified systematic variance
pc1_pct = pc_var_share[0]
pc12_pct = pc_var_share[:2].sum()
idiosync_pct = 100 - pc_var_share[:n_pcs_90-1].sum()

print(f"\n  {'PC':<6}  {'Var Contribution':>17}  {'% of Portfolio Var':>19}")
print(f"  {'-'*48}")
for i in range(N):
    bar = "█" * int(pc_var_share[i] / 2)
    print(f"  PC{i+1:<4}  {pc_var_contrib[i]:>17.6f}  {pc_var_share[i]:>18.2f}%  {bar}")

print(f"\n  Systematic (PC1 — market factor) : {pc1_pct:.2f}% of portfolio variance")
print(f"  PC1 + PC2 combined               : {pc12_pct:.2f}% of portfolio variance")
print(f"  Top {n_pcs_90} PCs (90% of return var)   : {pc_var_share[:n_pcs_90].sum():.2f}% of portfolio variance")
print(f"\n  INSIGHT: {pc1_pct:.1f}% of portfolio risk is pure Nifty market beta (PC1). "
      f"No stock-picking or sector rotation adds diversification against this systemic driver. "
      f"Only ~{100-pc12_pct:.1f}% of risk comes from PC3+ (idiosyncratic + sector effects).")


# ============================================================
# SAVE CSV
# ============================================================
results_rows = []
for i, nm in enumerate(NAMES):
    results_rows.append({
        "stock"          : nm,
        "ann_vol_pct"    : round(indiv_vols[i] * 100, 4),
        "beta_portfolio" : round(beta_port[i], 6),
        "marginal_var"   : round(mvar[i], 6),
        "component_var_rs": round(cvar_rs[i], 2),
        "component_var_pct":round(cvar_share[i], 4),
        "pc1_loading"    : round(loadings.loc[nm, "PC1"], 6),
        "pc2_loading"    : round(loadings.loc[nm, "PC2"], 6),
        "pc3_loading"    : round(loadings.loc[nm, "PC3"], 6),
    })

pca_results = pd.DataFrame(results_rows)
pca_results.to_csv("pca_results.csv", index=False)
print("\n  pca_results.csv saved.")


# ============================================================
# PLOTS
# ============================================================

# ── Plot 1: Covariance Heatmap ─────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.zeros_like(cov_df.values, dtype=bool)
# Show full matrix (no mask) — it's symmetric but readable both halves
sns.heatmap(
    cov_df * 100,        # display as % units
    annot  = True,
    fmt    = ".3f",
    cmap   = "RdYlGn_r",
    center = 0,
    ax     = ax,
    annot_kws = {"size": 7},
    cbar_kws  = {"label": "Annualised Covariance (×100)"},
    linewidths = 0.4,
)
ax.set_title("Annualised Return Covariance Matrix — Nifty 50 Portfolio",
             fontsize=13, fontweight="bold", pad=12)
ax.tick_params(axis="x", rotation=40)
ax.tick_params(axis="y", rotation=0)
plt.tight_layout()
plt.savefig("covariance_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  covariance_heatmap.png saved.")

# ── Plot 2: Component VaR bar chart ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

order = np.argsort(-cvar_share)
colors_bar = ["#d62728" if s > 12 else "#1f77b4" for s in cvar_share[order]]

ax = axes[0]
bars = ax.bar(np.array(NAMES)[order], cvar_rs[order] / 1e5, color=colors_bar,
              edgecolor="white", linewidth=0.5)
ax.set_xlabel("Stock", fontsize=11)
ax.set_ylabel("Component VaR (₹ Lakh)", fontsize=11)
ax.set_title("Component VaR by Stock  (99% 1-Day)", fontsize=12, fontweight="bold")
ax.tick_params(axis="x", rotation=40)
for bar, val in zip(bars, cvar_share[order]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=7.5)
ax.axhline(cvar_rs.mean()/1e5, color="grey", lw=1.2, ls="--",
           label=f"Avg ₹{cvar_rs.mean()/1e5:.2f}L")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.35)

# Pie chart of % contribution
ax2 = axes[1]
wedge_colors = plt.cm.tab10(np.linspace(0, 1, N))
wedges, texts, autotexts = ax2.pie(
    cvar_share,
    labels     = NAMES,
    autopct    = "%1.1f%%",
    colors     = wedge_colors,
    startangle = 140,
    pctdistance= 0.78,
    textprops  = {"fontsize": 8},
)
for a in autotexts:
    a.set_fontsize(7)
ax2.set_title("% Risk Contribution per Stock", fontsize=12, fontweight="bold")

plt.suptitle(f"Component VaR Decomposition — ₹{NOTIONAL/1e7:.0f}Cr Portfolio  "
             f"(Total 99% VaR = ₹{port_VaR_rs/1e5:.2f}L)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("component_var_chart.png", dpi=150, bbox_inches="tight")
plt.close()
print("  component_var_chart.png saved.")

# ── Plot 3: PCA Scree Chart ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.bar(range(1, N+1), expl_var, color="#1f77b4", edgecolor="white",
       linewidth=0.5, label="Individual")
ax.set_xlabel("Principal Component", fontsize=11)
ax.set_ylabel("Variance Explained (%)", fontsize=11)
ax.set_title("Scree Plot — Variance Explained by PC", fontsize=12, fontweight="bold")
ax.set_xticks(range(1, N+1))
ax.set_xticklabels([f"PC{i}" for i in range(1, N+1)])
for j, v in enumerate(expl_var):
    ax.text(j + 1, v + 0.3, f"{v:.1f}%", ha="center", fontsize=8)
ax.grid(axis="y", alpha=0.35)

ax2 = axes[1]
ax2.plot(range(1, N+1), cum_expl_var, "o-", color="#d62728", lw=2.2,
         ms=7, markerfacecolor="white", markeredgewidth=2)
ax2.axhline(90, color="grey", lw=1.2, ls="--", label="90% threshold")
ax2.axvline(n_pcs_90, color="orange", lw=1.2, ls="--",
            label=f"PC{n_pcs_90} = 90%")
ax2.fill_between(range(1, N+1), cum_expl_var, alpha=0.15, color="#d62728")
ax2.set_xlabel("Number of Principal Components", fontsize=11)
ax2.set_ylabel("Cumulative Variance Explained (%)", fontsize=11)
ax2.set_title("Cumulative Variance Explained", fontsize=12, fontweight="bold")
ax2.set_xticks(range(1, N+1))
ax2.legend(fontsize=9)
ax2.grid(alpha=0.35)
ax2.set_ylim(0, 105)

plt.suptitle("PCA on Nifty 50 Return Correlation Matrix",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("pca_scree.png", dpi=150, bbox_inches="tight")
plt.close()
print("  pca_scree.png saved.")

# ── Plot 4: PCA Loadings Heatmap (PC1–PC4) ─────────────────
fig, ax = plt.subplots(figsize=(10, 6))
load_plot = loadings.iloc[:, :4].copy()

# Normalise sign: PC1 should be all positive (market factor)
for col in load_plot.columns:
    if load_plot[col].mean() < 0:
        load_plot[col] = -load_plot[col]

ann_load = load_plot.applymap(lambda x: f"{x:.3f}")
sns.heatmap(
    load_plot,
    annot      = ann_load,
    fmt        = "",
    cmap       = "RdBu_r",
    center     = 0,
    vmin       = -0.7, vmax = 0.7,
    ax         = ax,
    linewidths = 0.5,
    annot_kws  = {"size": 9},
    cbar_kws   = {"label": "Factor Loading"},
)
labels = {
    "PC1": f"PC1 ({expl_var[0]:.1f}%) Market Beta",
    "PC2": f"PC2 ({expl_var[1]:.1f}%) Sector Rotation",
    "PC3": f"PC3 ({expl_var[2]:.1f}%) Sub-sector",
    "PC4": f"PC4 ({expl_var[3]:.1f}%) Residual",
}
ax.set_xticklabels([labels.get(c, c) for c in load_plot.columns], rotation=20, ha="right")
ax.set_title("PCA Factor Loadings — Top 4 Principal Components",
             fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Stock", fontsize=11)
plt.tight_layout()
plt.savefig("pca_loadings.png", dpi=150, bbox_inches="tight")
plt.close()
print("  pca_loadings.png saved.")

print("\n" + "=" * 65)
print("  ALL DONE.")
print("=" * 65)

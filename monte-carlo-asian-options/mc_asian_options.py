"""
Monte Carlo Asian Option Pricing Engine — Nifty50
===================================================
European MC vs BS  |  Antithetic + Control Variates
Asian Arithmetic   |  Convergence Analysis  |  Path Vis
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
import time

np.random.seed(42)

# ─────────────────────────────────────────────────────────
# MARKET PARAMETERS
# ─────────────────────────────────────────────────────────
S0    = 23_500.0
K     = 23_500.0
T     = 30 / 365
r     = 0.065
sigma = 0.16
M     = 252                        # trading days/year
N_MC  = 10_000                     # base simulation paths
STEPS = int(T * M)                 # ~8 daily steps in 30 days
# Weekly averaging: every 5 trading days
AVG_FREQ  = 5
AVG_TIMES = list(range(AVG_FREQ, STEPS + 1, AVG_FREQ))  # [5] for 30d
if STEPS not in AVG_TIMES:
    AVG_TIMES.append(STEPS)
AVG_TIMES = sorted(set(AVG_TIMES))

dt = T / STEPS

# ─────────────────────────────────────────────────────────
# SECTION HEADER HELPER
# ─────────────────────────────────────────────────────────
def header(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ============================================================
# STEP 1 — BLACK-SCHOLES ANALYTICAL
# ============================================================
def bs_call(S, K, r, T, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


BS_PRICE = bs_call(S0, K, r, T, sigma)


# ============================================================
# GBM PATH GENERATOR — vectorised
# ============================================================
def gbm_paths(n_paths, n_steps, antithetic=False, seed=None):
    """
    Returns paths array shape (n_paths, n_steps+1).
    If antithetic=True, returns (2*n_paths, n_steps+1) where second
    half uses negated draws.
    """
    rng = np.random.default_rng(seed)
    Z   = rng.standard_normal((n_paths, n_steps))
    log_increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    if antithetic:
        log_inc_anti = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * (-Z)
        log_increments = np.vstack([log_increments, log_inc_anti])

    log_paths = np.hstack([
        np.zeros((log_increments.shape[0], 1)),
        np.cumsum(log_increments, axis=1)
    ])
    return S0 * np.exp(log_paths)


# ============================================================
# STEP 1 — STANDARD EUROPEAN MC
# ============================================================
header("STEP 1 — STANDARD EUROPEAN MC vs BLACK-SCHOLES")

paths   = gbm_paths(N_MC, STEPS, seed=0)
S_T     = paths[:, -1]
payoffs = np.maximum(S_T - K, 0) * np.exp(-r * T)

mc_price    = payoffs.mean()
mc_stderr   = payoffs.std() / np.sqrt(N_MC)
mc_ci_lo    = mc_price - 1.96 * mc_stderr
mc_ci_hi    = mc_price + 1.96 * mc_stderr

print(f"\n  Black-Scholes Analytical : ₹{BS_PRICE:>10.4f}")
print(f"  MC Standard (N={N_MC:,})   : ₹{mc_price:>10.4f}")
print(f"  Std Error                : ₹{mc_stderr:>10.4f}")
print(f"  95% CI                   : [₹{mc_ci_lo:.4f}, ₹{mc_ci_hi:.4f}]")
print(f"  CI Width                 :  ₹{mc_ci_hi - mc_ci_lo:.4f}")
print(f"  Error vs BS              :  ₹{abs(mc_price - BS_PRICE):.4f}  "
      f"({abs(mc_price - BS_PRICE)/BS_PRICE*100:.3f}%)")


# ============================================================
# STEP 2 — ANTITHETIC VARIATES
# ============================================================
header("STEP 2 — ANTITHETIC VARIATES")

paths_av = gbm_paths(N_MC, STEPS, antithetic=True, seed=0)
S_T_av   = paths_av[:, -1]
pay_av   = np.maximum(S_T_av - K, 0) * np.exp(-r * T)

# Pair original and antithetic
pay_orig = pay_av[:N_MC]
pay_anti = pay_av[N_MC:]
pay_paired = (pay_orig + pay_anti) / 2          # paired average

av_price  = pay_paired.mean()
av_stderr = pay_paired.std() / np.sqrt(N_MC)
av_ci_lo  = av_price - 1.96 * av_stderr
av_ci_hi  = av_price + 1.96 * av_stderr

var_std = np.var(payoffs)
var_av  = np.var(pay_paired)
vr_av   = var_std / var_av

print(f"\n  Antithetic Price (N={N_MC:,}): ₹{av_price:>10.4f}")
print(f"  Std Error                : ₹{av_stderr:>10.4f}")
print(f"  95% CI                   : [₹{av_ci_lo:.4f}, ₹{av_ci_hi:.4f}]")
print(f"  CI Width                 :  ₹{av_ci_hi - av_ci_lo:.4f}")
print(f"  Variance Std MC          :  {var_std:.6f}")
print(f"  Variance Antithetic      :  {var_av:.6f}")
print(f"  Variance Reduction Ratio :  {vr_av:.2f}x  "
      f"(antithetic {(1-1/vr_av)*100:.1f}% lower variance)")
print(f"  CI Width Reduction       :  {(1-(av_ci_hi-av_ci_lo)/(mc_ci_hi-mc_ci_lo))*100:.1f}%")


# ============================================================
# GEOMETRIC ASIAN CLOSED-FORM (control variate baseline)
# ============================================================
def geo_asian_call_cf(S, K, r, T, sigma, n_avg):
    """
    Kemna-Vorst (1990) discrete geometric Asian call.
    n_avg equally-spaced obs at T/n, 2T/n, ..., T.
    sigma_g = sigma * sqrt((n+1)(2n+1) / (6n^2))
    mu_g    = (r - sigma^2/2) * (n+1) / (2n)
    """
    sigma_g = sigma * np.sqrt((n_avg + 1) * (2 * n_avg + 1) / (6 * n_avg**2))
    mu_g    = (r - 0.5 * sigma**2) * (n_avg + 1) / (2 * n_avg)
    d1      = (np.log(S / K) + (mu_g + 0.5 * sigma_g**2) * T) / (sigma_g * np.sqrt(T))
    d2      = d1 - sigma_g * np.sqrt(T)
    return np.exp(-r * T) * (S * np.exp(mu_g * T) * norm.cdf(d1) - K * norm.cdf(d2))


N_AVG      = len(AVG_TIMES)
GEO_CF     = geo_asian_call_cf(S0, K, r, T, sigma, N_AVG)


# ============================================================
# ASIAN PAYOFF HELPER
# ============================================================
def asian_payoffs(paths, avg_times):
    """Arithmetic and geometric Asian payoffs from full paths."""
    avg_pts    = paths[:, avg_times]                   # shape (N, n_avg)
    arith_avg  = avg_pts.mean(axis=1)
    geo_avg    = np.exp(np.log(avg_pts).mean(axis=1))
    pay_arith  = np.maximum(arith_avg - K, 0) * np.exp(-r * T)
    pay_geo    = np.maximum(geo_avg   - K, 0) * np.exp(-r * T)
    return pay_arith, pay_geo


# ============================================================
# STEP 3 — CONTROL VARIATES
# ============================================================
header("STEP 3 — CONTROL VARIATES (Geometric Asian as CV)")

paths_cv            = gbm_paths(N_MC, STEPS, seed=1)
pay_arith_cv, pay_geo_cv = asian_payoffs(paths_cv, AVG_TIMES)

# OLS: regress arithmetic on geometric → optimal beta
cov_mat    = np.cov(pay_arith_cv, pay_geo_cv)
beta_opt   = cov_mat[0, 1] / cov_mat[1, 1]
rho_corr   = cov_mat[0, 1] / np.sqrt(cov_mat[0, 0] * cov_mat[1, 1])

# Control-variate adjusted estimator
pay_cv_adj = pay_arith_cv - beta_opt * (pay_geo_cv - GEO_CF)

cv_price   = pay_cv_adj.mean()
cv_stderr  = pay_cv_adj.std() / np.sqrt(N_MC)
cv_ci_lo   = cv_price - 1.96 * cv_stderr
cv_ci_hi   = cv_price + 1.96 * cv_stderr

# Variance reduction
var_arith_std = np.var(pay_arith_cv)
var_cv_adj    = np.var(pay_cv_adj)
vr_cv         = var_arith_std / var_cv_adj

print(f"\n  Geometric Asian CF price : ₹{GEO_CF:>10.4f}")
print(f"  Averaging points         :  {N_AVG} ({AVG_TIMES})")
print(f"  Optimal beta (OLS)       :  {beta_opt:.6f}")
print(f"  Correlation (arith,geo)  :  {rho_corr:.6f}")
print(f"  Var Std Asian MC         :  {var_arith_std:.8f}")
print(f"  Var Control-Variate      :  {var_cv_adj:.8f}")
print(f"  Variance Reduction Ratio :  {vr_cv:.2f}x")
print(f"  Theoretical max (1-rho²) :  {1/(1-rho_corr**2):.2f}x")


# ============================================================
# STEP 4 — ASIAN OPTION PRICING COMPARISON
# ============================================================
header("STEP 4 — ARITHMETIC ASIAN CALL — ALL THREE METHODS")

# Standard MC Asian
paths_std      = gbm_paths(N_MC, STEPS, seed=2)
pay_std_asian, pay_geo_std = asian_payoffs(paths_std, AVG_TIMES)
asian_std_price  = pay_std_asian.mean()
asian_std_se     = pay_std_asian.std() / np.sqrt(N_MC)

# Antithetic Asian
paths_anti_full = gbm_paths(N_MC, STEPS, antithetic=True, seed=2)
pay_anti_a, _   = asian_payoffs(paths_anti_full[:N_MC], AVG_TIMES)
pay_anti_b, _   = asian_payoffs(paths_anti_full[N_MC:], AVG_TIMES)
pay_anti_asian  = (pay_anti_a + pay_anti_b) / 2
asian_av_price  = pay_anti_asian.mean()
asian_av_se     = pay_anti_asian.std() / np.sqrt(N_MC)

# Control Variate Asian
pay_cv_a, pay_geo_a = asian_payoffs(paths_std, AVG_TIMES)
cov2      = np.cov(pay_cv_a, pay_geo_a)
beta2     = cov2[0, 1] / cov2[1, 1]
pay_cv2   = pay_cv_a - beta2 * (pay_geo_a - GEO_CF)
asian_cv_price = pay_cv2.mean()
asian_cv_se    = pay_cv2.std() / np.sqrt(N_MC)

def ci(price, se):
    return price - 1.96*se, price + 1.96*se

print(f"\n  {'Method':<26} {'Price (₹)':>10}  {'StdErr':>8}  {'CI Low':>10}  {'CI High':>10}  {'CI Width':>9}")
print(f"  {'-'*80}")
for name, p, se in [
    ("Standard MC",          asian_std_price, asian_std_se),
    ("Antithetic Variates",  asian_av_price,  asian_av_se),
    ("Control Variates (CV)",asian_cv_price,  asian_cv_se),
    ("Geometric CF (exact)", GEO_CF,          0),
]:
    lo, hi = ci(p, se)
    w = hi - lo
    print(f"  {name:<26} {p:>10.4f}  {se:>8.4f}  {lo:>10.4f}  {hi:>10.4f}  {w:>9.4f}")

vr_av_asian = np.var(pay_std_asian) / np.var(pay_anti_asian)
vr_cv_asian = np.var(pay_std_asian) / np.var(pay_cv2)
print(f"\n  Variance Reduction — Antithetic : {vr_av_asian:.2f}x")
print(f"  Variance Reduction — Control V  : {vr_cv_asian:.2f}x")


# ============================================================
# STEP 5 — CONVERGENCE ANALYSIS
# ============================================================
header("STEP 5 — CONVERGENCE ANALYSIS")

N_VALS = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]

conv_rows = []
print(f"\n  {'N':>8}  {'Std Price':>10}  {'Std Err':>10}  {'Anti Price':>11}  {'Anti Err':>11}")
print(f"  {'-'*58}")

for n in N_VALS:
    # Standard
    p_std = gbm_paths(n, STEPS, antithetic=False, seed=7)
    payoff_s = np.maximum(p_std[:, -1] - K, 0) * np.exp(-r * T)
    pr_s  = payoff_s.mean()
    err_s = abs(pr_s - BS_PRICE)
    se_s  = payoff_s.std() / np.sqrt(n)

    # Antithetic
    p_av  = gbm_paths(n, STEPS, antithetic=True, seed=7)
    pay_o = np.maximum(p_av[:n, -1] - K, 0) * np.exp(-r * T)
    pay_a = np.maximum(p_av[n:, -1] - K, 0) * np.exp(-r * T)
    pay_p = (pay_o + pay_a) / 2
    pr_a  = pay_p.mean()
    err_a = abs(pr_a - BS_PRICE)
    se_a  = pay_p.std() / np.sqrt(n)

    conv_rows.append({
        "N": n, "std_price": pr_s, "std_error": err_s, "std_se": se_s,
        "anti_price": pr_a, "anti_error": err_a, "anti_se": se_a
    })
    print(f"  {n:>8,}  {pr_s:>10.4f}  {err_s:>10.5f}  {pr_a:>11.4f}  {err_a:>11.5f}")

conv_df = pd.DataFrame(conv_rows)
print(f"\n  BS Analytical Price: ₹{BS_PRICE:.4f}")


# ============================================================
# RESULTS SUMMARY TABLE
# ============================================================
header("FULL RESULTS SUMMARY")

print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │  EUROPEAN CALL  (K={K:.0f}, T=30d, r={r*100:.1f}%, σ={sigma*100:.0f}%)         │
  ├─────────────────────────────────────────────────────────────┤
  │  Black-Scholes Exact        : ₹{BS_PRICE:>9.4f}                    │
  │  Standard MC  (N=10k)       : ₹{mc_price:>9.4f}  ±{mc_stderr:.4f}           │
  │  Antithetic   (N=10k)       : ₹{av_price:>9.4f}  ±{av_stderr:.4f}           │
  │  Var Reduction (Antithetic) : {vr_av:>6.2f}x                        │
  ├─────────────────────────────────────────────────────────────┤
  │  ARITHMETIC ASIAN CALL  (weekly avg, {N_AVG} points)              │
  ├─────────────────────────────────────────────────────────────┤
  │  Standard MC  (N=10k)       : ₹{asian_std_price:>9.4f}  ±{asian_std_se:.4f}           │
  │  Antithetic   (N=10k)       : ₹{asian_av_price:>9.4f}  ±{asian_av_se:.4f}           │
  │  Control Var  (N=10k)       : ₹{asian_cv_price:>9.4f}  ±{asian_cv_se:.4f}           │
  │  Geometric CF (exact)       : ₹{GEO_CF:>9.4f}                    │
  │  Var Reduction (Antithetic) : {vr_av_asian:>6.2f}x                        │
  │  Var Reduction (Control V.) : {vr_cv_asian:>6.2f}x                        │
  └─────────────────────────────────────────────────────────────┘""")


# ============================================================
# SAVE CSV
# ============================================================
asian_csv = pd.DataFrame([
    {"method": "European BS (exact)",    "price": BS_PRICE,        "std_error": 0,           "ci_lo": BS_PRICE,        "ci_hi": BS_PRICE,        "var_reduction": 1.0},
    {"method": "European MC Std",        "price": mc_price,        "std_error": mc_stderr,   "ci_lo": mc_ci_lo,        "ci_hi": mc_ci_hi,        "var_reduction": 1.0},
    {"method": "European MC Antithetic", "price": av_price,        "std_error": av_stderr,   "ci_lo": av_ci_lo,        "ci_hi": av_ci_hi,        "var_reduction": vr_av},
    {"method": "Asian Arith Std MC",     "price": asian_std_price, "std_error": asian_std_se,"ci_lo": ci(asian_std_price,asian_std_se)[0], "ci_hi": ci(asian_std_price,asian_std_se)[1], "var_reduction": 1.0},
    {"method": "Asian Arith Antithetic", "price": asian_av_price,  "std_error": asian_av_se, "ci_lo": ci(asian_av_price,asian_av_se)[0],  "ci_hi": ci(asian_av_price,asian_av_se)[1],  "var_reduction": vr_av_asian},
    {"method": "Asian Arith Control V.", "price": asian_cv_price,  "std_error": asian_cv_se, "ci_lo": ci(asian_cv_price,asian_cv_se)[0],  "ci_hi": ci(asian_cv_price,asian_cv_se)[1],  "var_reduction": vr_cv_asian},
    {"method": "Asian Geometric CF",     "price": GEO_CF,          "std_error": 0,           "ci_lo": GEO_CF,          "ci_hi": GEO_CF,          "var_reduction": None},
])
asian_csv = asian_csv.round(6)
asian_csv.to_csv("asian_option_results.csv", index=False)
print("\n  asian_option_results.csv saved.")


# ============================================================
# STEP 6 — PLOTS
# ============================================================

# ── Plot A: GBM Paths + Terminal Distribution ──────────────
fig = plt.figure(figsize=(14, 5))
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

ax1 = fig.add_subplot(gs[0])
paths_vis = gbm_paths(50, STEPS, seed=99)
t_grid    = np.linspace(0, 30, STEPS + 1)
for i in range(50):
    ax1.plot(t_grid, paths_vis[i], lw=0.8, alpha=0.55,
             color=plt.cm.plasma(i / 50))
ax1.axhline(K, color="crimson", lw=1.5, ls="--", label=f"Strike ₹{K:,.0f}")
ax1.axhline(S0, color="steelblue", lw=1.2, ls=":", label=f"Spot ₹{S0:,.0f}")
ax1.set_xlabel("Days", fontsize=11)
ax1.set_ylabel("Nifty Level", fontsize=11)
ax1.set_title("50 Simulated GBM Paths (30-day)", fontsize=12, fontweight="bold")
ax1.legend(fontsize=9)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
ax1.grid(alpha=0.3)

ax2 = fig.add_subplot(gs[1])
S_T_large = gbm_paths(50_000, STEPS, seed=5)[:, -1]
ax2.hist(S_T_large, bins=80, color="steelblue", alpha=0.75,
         edgecolor="white", linewidth=0.3, density=True)
ax2.axvline(K,  color="crimson",   lw=2, ls="--", label=f"Strike/Spot ₹{K:,.0f}")
ax2.axvline(S_T_large.mean(), color="darkorange", lw=2,
            label=f"Mean ₹{S_T_large.mean():,.0f}")
ax2.set_xlabel("Terminal Nifty Level S_T", fontsize=11)
ax2.set_ylabel("Density", fontsize=11)
ax2.set_title("Terminal Price Distribution (N=50k)", fontsize=12, fontweight="bold")
ax2.legend(fontsize=9)
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
ax2.grid(alpha=0.3)

plt.suptitle("Nifty GBM Simulation — Monte Carlo Asian Options",
             fontsize=13, fontweight="bold", y=1.02)
plt.savefig("gbm_paths.png", dpi=150, bbox_inches="tight")
plt.close()
print("  gbm_paths.png saved.")


# ── Plot B: Convergence — log-log error vs N ──────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ns     = conv_df["N"].values
e_std  = conv_df["std_error"].values
e_anti = conv_df["anti_error"].values

# Theoretical N^{-1/2} reference lines anchored at N=100
ref_x = np.array([100, 100_000])
scale_s = e_std[0] * ref_x[0]**0.5
scale_a = e_anti[0] * ref_x[0]**0.5
ref_std  = scale_s / ref_x**0.5
ref_anti = scale_a / ref_x**0.5

ax = axes[0]
ax.loglog(ns, e_std,  "o-", color="#1f77b4", lw=2,   ms=7, label="Standard MC")
ax.loglog(ns, e_anti, "s-", color="#d62728", lw=2,   ms=7, label="Antithetic")
ax.loglog(ref_x, ref_std,  "--", color="#1f77b4", alpha=0.5, lw=1.2, label="O(N⁻⁰·⁵) ref Std")
ax.loglog(ref_x, ref_anti, "--", color="#d62728", alpha=0.5, lw=1.2, label="O(N⁻⁰·⁵) ref Anti")
ax.set_xlabel("Number of Paths (N)", fontsize=11)
ax.set_ylabel("|Price Error| vs BS  (₹)", fontsize=11)
ax.set_title("Convergence — Absolute Error vs N", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, which="both", alpha=0.3)

ax2 = axes[1]
se_std  = conv_df["std_se"].values
se_anti = conv_df["anti_se"].values
ax2.loglog(ns, se_std,  "o-", color="#1f77b4", lw=2, ms=7, label="Standard MC SE")
ax2.loglog(ns, se_anti, "s-", color="#d62728", lw=2, ms=7, label="Antithetic SE")
# Annotate variance reduction at N=10k
n10k_idx = list(ns).index(10_000)
ax2.annotate(
    f"VR={vr_av:.1f}x\n@N=10k",
    xy=(10_000, se_anti[n10k_idx]),
    xytext=(3_000, se_anti[n10k_idx] * 2.5),
    arrowprops=dict(arrowstyle="->", color="grey"),
    fontsize=9, color="#d62728"
)
ax2.set_xlabel("Number of Paths (N)", fontsize=11)
ax2.set_ylabel("Standard Error of Estimate (₹)", fontsize=11)
ax2.set_title("Standard Error Convergence", fontsize=12, fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(True, which="both", alpha=0.3)

plt.suptitle("Monte Carlo Convergence — Standard vs Antithetic Variates",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("convergence_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print("  convergence_plot.png saved.")

print("\n" + "=" * 65)
print("  ALL DONE.")
print("=" * 65)

"""
Nifty Options Implied Volatility Surface
=========================================
Synthetic option chain → BS pricing → IV extraction → SVI fit → plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq, minimize
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# MARKET PARAMETERS
# ─────────────────────────────────────────────
SPOT    = 23_500.0
STRIKES = np.arange(21_000, 26_500, 500)          # 21000 … 26000  (11 strikes)
TAUS    = np.array([7, 14, 30, 45, 60, 90]) / 365 # in years
RATE    = 0.065                                    # RBI repo rate

# ─────────────────────────────────────────────
# STEP 0 — SYNTHETIC MARKET DATA
# ─────────────────────────────────────────────

def synthetic_iv(strike, tau_days, spot=SPOT):
    """
    Heston-inspired IV smile: base vol + smile + skew + term-structure tilt.
    - smile  : OTM options costlier (quadratic in log-moneyness)
    - skew   : puts (low strikes) more expensive than calls (negative skew)
    - term   : short-dated vol > long-dated vol (inverted term structure)
    """
    k   = np.log(strike / spot)         # log-moneyness
    tau = tau_days / 365.0

    # Term structure: vol decays from ~18% at 7d to ~14% at 90d
    base_vol  = 0.18 - 0.04 * (1 - np.exp(-tau * 6))

    # Smile (symmetric upward curve from ATM)
    smile     = 0.35 * k**2

    # Skew: puts more expensive, linear tilt
    skew      = -0.25 * k

    iv = base_vol + smile + skew
    # Floor/cap to stay realistic
    return float(np.clip(iv, 0.08, 0.80))


# Pre-build IV matrix  shape: (n_strikes, n_taus)
IV_TRUE = np.array([
    [synthetic_iv(K, t * 365) for t in TAUS]
    for K in STRIKES
])


# ─────────────────────────────────────────────
# STEP 1 — BLACK-SCHOLES PRICER + GREEKS
# ─────────────────────────────────────────────

def _d1d2(S, K, r, T, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(S, K, r, T, sigma, flag="call"):
    d1, d2 = _d1d2(S, K, r, T, sigma)
    if flag == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_greeks(S, K, r, T, sigma, flag="call"):
    d1, d2 = _d1d2(S, K, r, T, sigma)
    nd1    = norm.pdf(d1)
    Nd1    = norm.cdf(d1)
    Nd2    = norm.cdf(d2)

    delta = Nd1 if flag == "call" else Nd1 - 1
    gamma = nd1 / (S * sigma * np.sqrt(T))
    vega  = S * nd1 * np.sqrt(T) / 100          # per 1% move in vol
    if flag == "call":
        theta = (-(S * nd1 * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * Nd2) / 365
        rho   =  K * T * np.exp(-r * T) * Nd2  / 100
    else:
        theta = (-(S * nd1 * sigma) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        rho   = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return {"delta": delta, "gamma": gamma, "vega": vega,
            "theta": theta, "rho": rho}


# ─────────────────────────────────────────────
# PUT-CALL PARITY VERIFICATION
# ─────────────────────────────────────────────

print("=" * 60)
print("  PUT-CALL PARITY VERIFICATION (30-day slice)")
print("=" * 60)
T30   = 30 / 365
sig30 = synthetic_iv(SPOT, 30)
pcp_errors = []
for K in STRIKES:
    c = bs_price(SPOT, K, RATE, T30, sig30, "call")
    p = bs_price(SPOT, K, RATE, T30, sig30, "put")
    parity_lhs = c - p
    parity_rhs = SPOT - K * np.exp(-RATE * T30)
    pcp_errors.append(abs(parity_lhs - parity_rhs))

print(f"  Max put-call parity error: {max(pcp_errors):.2e}  ✓" if max(pcp_errors) < 1e-8
      else f"  Max put-call parity error: {max(pcp_errors):.2e}  ✗")
print()


# ─────────────────────────────────────────────
# STEP 2 — IMPLIED VOLATILITY EXTRACTION (Brent)
# ─────────────────────────────────────────────

def implied_vol(market_price, S, K, r, T, flag="call"):
    intrinsic = max(S - K, 0) if flag == "call" else max(K - S, 0)
    disc_fwd  = S - K * np.exp(-r * T) if flag == "call" else K * np.exp(-r * T) - S
    if market_price <= intrinsic or market_price <= 0:
        return np.nan
    # Upper bound: vol so large that price ≈ intrinsic + forward disc
    try:
        iv = brentq(
            lambda sig: bs_price(S, K, r, T, sig, flag) - market_price,
            1e-6, 10.0,
            xtol=1e-8, maxiter=200
        )
        return iv
    except ValueError:
        return np.nan


# Build synthetic market prices, then back out IV
IV_SURFACE = np.full((len(STRIKES), len(TAUS)), np.nan)
PRICES_CALL = np.full_like(IV_SURFACE, np.nan)
PRICES_PUT  = np.full_like(IV_SURFACE, np.nan)

for i, K in enumerate(STRIKES):
    for j, T in enumerate(TAUS):
        iv_true = IV_TRUE[i, j]
        c_price = bs_price(SPOT, K, RATE, T, iv_true, "call")
        p_price = bs_price(SPOT, K, RATE, T, iv_true, "put")
        PRICES_CALL[i, j] = c_price
        PRICES_PUT[i, j]  = p_price

        # Use call for ITM/ATM/OTM calls, put for deep ITM puts
        flag   = "put" if K < SPOT * 0.97 else "call"
        mkt_px = p_price if flag == "put" else c_price
        iv_ext = implied_vol(mkt_px, SPOT, K, RATE, T, flag)
        IV_SURFACE[i, j] = iv_ext


# ─────────────────────────────────────────────
# STEP 3 — SVI FIT PER MATURITY SLICE
# ─────────────────────────────────────────────

def svi_variance(k, a, b, rho, m, sigma):
    """SVI total variance w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))"""
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))


def fit_svi(log_moneyness, iv_slice):
    """Fit SVI params to one maturity slice via least-squares."""
    mask = ~np.isnan(iv_slice)
    k    = log_moneyness[mask]
    w    = (iv_slice[mask])**2          # total implied variance

    def residuals(params):
        a, b, rho, m, sigma = params
        w_fit = svi_variance(k, a, b, rho, m, sigma)
        return np.sum((w - w_fit)**2)

    bounds = [(-0.5, 0.5), (1e-4, 2.0), (-0.999, 0.999), (-1.0, 1.0), (1e-4, 2.0)]
    p0     = [0.04, 0.1, -0.3, 0.0, 0.1]

    res = minimize(residuals, p0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 5000, "ftol": 1e-14})
    return res.x if res.success else p0


log_moneyness = np.log(STRIKES / SPOT)
svi_params    = {}

print("=" * 60)
print("  SVI FIT — PARAMETERS PER MATURITY")
print("=" * 60)
print(f"  {'Days':>5}  {'a':>8}  {'b':>8}  {'rho':>8}  {'m':>8}  {'sigma':>8}")
for j, T in enumerate(TAUS):
    tau_days = int(round(T * 365))
    params   = fit_svi(log_moneyness, IV_SURFACE[:, j])
    svi_params[tau_days] = params
    a, b, rho, m, sigma = params
    print(f"  {tau_days:>5}  {a:>8.5f}  {b:>8.5f}  {rho:>8.4f}  {m:>8.5f}  {sigma:>8.5f}")
print()


# ─────────────────────────────────────────────
# STEP 4 — ANALYSIS TABLES
# ─────────────────────────────────────────────

# ── 4a. IV Smile Table (30-day) ─────────────────

def moneyness_label(K, S):
    ratio = K / S
    if ratio < 0.90:  return "DOTM-P"
    if ratio < 0.97:  return " ITM-P"
    if ratio < 1.03:  return "   ATM"
    if ratio < 1.10:  return " ITM-C"
    return "DOTM-C"


T30_idx   = list(np.round(TAUS * 365).astype(int)).index(30)
iv_30d    = IV_SURFACE[:, T30_idx]

print("=" * 60)
print("  IV SMILE TABLE — 30-day Maturity")
print("=" * 60)
print(f"  {'Strike':>8}  {'Moneyness':>8}  {'Log-Mon':>8}  {'IV (%)':>8}  {'Call ₹':>10}  {'Put ₹':>10}")
for i, K in enumerate(STRIKES):
    iv  = iv_30d[i]
    lbl = moneyness_label(K, SPOT)
    lm  = log_moneyness[i]
    c   = PRICES_CALL[i, T30_idx]
    p   = PRICES_PUT[i, T30_idx]
    print(f"  {K:>8.0f}  {lbl:>8}  {lm:>8.4f}  {iv*100:>8.2f}  {c:>10.2f}  {p:>10.2f}")
print()

# ── 4b. Put Skew ────────────────────────────────

atm_idx  = np.argmin(np.abs(STRIKES - SPOT))
otm90_K  = SPOT * 0.90
otm90_idx = np.argmin(np.abs(STRIKES - otm90_K))

iv_atm   = iv_30d[atm_idx]
iv_90    = iv_30d[otm90_idx]
put_skew = iv_90 - iv_atm

print("=" * 60)
print("  SKEW ANALYSIS — 30-day Maturity")
print("=" * 60)
print(f"  ATM strike         : {STRIKES[atm_idx]:.0f}")
print(f"  ATM IV             : {iv_atm*100:.2f}%")
print(f"  90% strike         : {STRIKES[otm90_idx]:.0f}  (≈ {STRIKES[otm90_idx]/SPOT:.2%} moneyness)")
print(f"  90% strike IV      : {iv_90*100:.2f}%")
print(f"  Put skew (90%-ATM) : {put_skew*100:+.2f}%")
print()

# ── 4c. ATM Term Structure ───────────────────────

print("=" * 60)
print("  ATM IV TERM STRUCTURE")
print("=" * 60)
print(f"  {'Days':>6}  {'ATM IV (%)':>12}  {'Total Var':>10}")
for j, T in enumerate(TAUS):
    tau_days = int(round(T * 365))
    iv_atm_t = IV_SURFACE[atm_idx, j]
    tot_var  = iv_atm_t**2 * T
    print(f"  {tau_days:>6}  {iv_atm_t*100:>12.2f}  {tot_var:>10.5f}")
print()

# ── 4d. Greeks table (ATM, 30d) ─────────────────

T30     = 30 / 365
iv_atm30 = IV_SURFACE[atm_idx, T30_idx]
K_atm    = STRIKES[atm_idx]

call_greeks = bs_greeks(SPOT, K_atm, RATE, T30, iv_atm30, "call")
put_greeks  = bs_greeks(SPOT, K_atm, RATE, T30, iv_atm30, "put")

print("=" * 60)
print(f"  GREEKS — ATM (K={K_atm:.0f}), 30-day, IV={iv_atm30*100:.2f}%")
print("=" * 60)
print(f"  {'Greek':<12}  {'Call':>12}  {'Put':>12}  {'Note'}")
rows = [
    ("Delta",  call_greeks["delta"], put_greeks["delta"], "∂P/∂S"),
    ("Gamma",  call_greeks["gamma"], put_greeks["gamma"], "∂²P/∂S²"),
    ("Theta",  call_greeks["theta"], put_greeks["theta"], "per day (₹)"),
    ("Vega",   call_greeks["vega"],  put_greeks["vega"],  "per 1% Δvol (₹)"),
    ("Rho",    call_greeks["rho"],   put_greeks["rho"],   "per 1% Δrate (₹)"),
]
for name, cg, pg, note in rows:
    print(f"  {name:<12}  {cg:>12.5f}  {pg:>12.5f}  {note}")
print()


# ─────────────────────────────────────────────
# STEP 5 — CSV OUTPUT
# ─────────────────────────────────────────────

tau_days_list = [int(round(t * 365)) for t in TAUS]
rows_list = []
for i, K in enumerate(STRIKES):
    for j, T_days in enumerate(tau_days_list):
        rows_list.append({
            "strike"        : K,
            "maturity_days" : T_days,
            "log_moneyness" : round(log_moneyness[i], 6),
            "iv"            : round(IV_SURFACE[i, j], 6),
            "call_price"    : round(PRICES_CALL[i, j], 4),
            "put_price"     : round(PRICES_PUT[i, j], 4),
        })

df = pd.DataFrame(rows_list)
df.to_csv("iv_surface_data.csv", index=False)
print("  iv_surface_data.csv saved.\n")


# ─────────────────────────────────────────────
# STEP 5 — PLOTS
# ─────────────────────────────────────────────

# ── Plot 1: 3-D IV Surface ───────────────────────

fig = plt.figure(figsize=(12, 7))
ax  = fig.add_subplot(111, projection="3d")

X, Y = np.meshgrid(STRIKES, np.array(tau_days_list))
Z    = IV_SURFACE.T * 100   # shape: (n_taus, n_strikes)

surf = ax.plot_surface(X, Y, Z, cmap="RdYlGn_r", alpha=0.88,
                       edgecolor="k", linewidth=0.3)
ax.set_xlabel("Strike", labelpad=10)
ax.set_ylabel("Days to Expiry", labelpad=10)
ax.set_zlabel("Implied Vol (%)", labelpad=10)
ax.set_title("Nifty Options — Implied Volatility Surface", fontsize=14, fontweight="bold")
fig.colorbar(surf, ax=ax, shrink=0.5, label="IV (%)")
plt.tight_layout()
plt.savefig("iv_surface_3d.png", dpi=150, bbox_inches="tight")
plt.close()
print("  iv_surface_3d.png saved.")

# ── Plot 2: IV Smile — 30-day ────────────────────

k_fine  = np.linspace(log_moneyness[0] - 0.02, log_moneyness[-1] + 0.02, 300)
a, b, rho, m, sigma = svi_params[30]
w_svi   = svi_variance(k_fine, a, b, rho, m, sigma)
iv_svi  = np.sqrt(np.maximum(w_svi, 0)) * 100

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(k_fine, iv_svi, "b-", lw=2.2, label="SVI fit")
ax.scatter(log_moneyness, iv_30d * 100, color="crimson", zorder=5,
           s=60, label="Extracted IV")
ax.axvline(0, color="grey", lw=1, ls="--", label="ATM")
for lk, K in zip(log_moneyness, STRIKES):
    ax.annotate(f"{K}", (lk, iv_30d[np.where(STRIKES == K)[0][0]] * 100 + 0.15),
                ha="center", fontsize=7, color="black")
ax.set_xlabel("Log-Moneyness  ln(K/S)", fontsize=11)
ax.set_ylabel("Implied Volatility (%)", fontsize=11)
ax.set_title("Nifty 30-Day IV Smile  (SVI fit)", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.35)
plt.tight_layout()
plt.savefig("iv_smile_30d.png", dpi=150, bbox_inches="tight")
plt.close()
print("  iv_smile_30d.png saved.")

# ── Plot 3: ATM Term Structure ───────────────────

atm_ivs = IV_SURFACE[atm_idx, :] * 100

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(tau_days_list, atm_ivs, "o-", color="steelblue", lw=2.2,
        markersize=8, markerfacecolor="white", markeredgewidth=2)
for x, y in zip(tau_days_list, atm_ivs):
    ax.annotate(f"{y:.2f}%", (x, y + 0.12), ha="center", fontsize=9)
ax.set_xlabel("Days to Expiry", fontsize=11)
ax.set_ylabel("ATM Implied Volatility (%)", fontsize=11)
ax.set_title("Nifty ATM IV — Term Structure", fontsize=13, fontweight="bold")
ax.grid(alpha=0.35)
plt.tight_layout()
plt.savefig("iv_term_structure.png", dpi=150, bbox_inches="tight")
plt.close()
print("  iv_term_structure.png saved.")

print()
print("=" * 60)
print("  ALL DONE — IV Surface complete.")
print("=" * 60)

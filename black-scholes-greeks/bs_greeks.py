"""
Black-Scholes Options Pricer — Full Greeks Analysis
=====================================================
Nifty50 options: 9 strikes × 6 maturities
Steps 1-6: pricer, greeks, sensitivity, moneyness, interview Qs, plots
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────
S0      = 23_500.0
STRIKES = np.array([21000, 22000, 22500, 23000, 23500, 24000, 24500, 25000, 26000])
TAUS_D  = np.array([7, 14, 30, 45, 60, 90])          # days
TAUS    = TAUS_D / 365.0
RATE    = 0.065
SIGMA   = 0.16

def hdr(title):
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)


# ============================================================
# STEP 1 — BLACK-SCHOLES PRICER FROM SCRATCH
# ============================================================
hdr("STEP 1 — BLACK-SCHOLES PRICER FROM SCRATCH")

def bs_d1d2(S, K, r, T, sigma):
    """Core d1, d2 terms."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def bs_call(S, K, r, T, sigma):
    d1, d2 = bs_d1d2(S, K, r, T, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put(S, K, r, T, sigma):
    d1, d2 = bs_d1d2(S, K, r, T, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# ── Put-call parity verification ──────────────────────────
print("\n  PUT-CALL PARITY VERIFICATION  (C - P = S - K·e^{-rT})")
print(f"  {'Strike':>8}  {'T(days)':>8}  {'C - P':>12}  {'S - Ke^-rT':>12}  {'Error':>14}")
print(f"  {'-'*62}")
max_err = 0.0
for K in STRIKES:
    for T in TAUS:
        C   = bs_call(S0, K, RATE, T, SIGMA)
        P   = bs_put(S0, K, RATE, T, SIGMA)
        lhs = C - P
        rhs = S0 - K * np.exp(-RATE * T)
        err = abs(lhs - rhs)
        max_err = max(max_err, err)
        if T == 30/365 and K in [22000, 23500, 25000]:   # print a sample
            print(f"  {K:>8.0f}  {int(T*365):>8}  {lhs:>12.6f}  {rhs:>12.6f}  {err:>14.2e}")

print(f"\n  Max parity error across all {len(STRIKES)*len(TAUS)} contracts: {max_err:.2e}  "
      f"{'✓' if max_err < 1e-10 else '✗'}")

# Sample prices at ATM 30d
T30 = 30 / 365
c_atm = bs_call(S0, S0, RATE, T30, SIGMA)
p_atm = bs_put(S0, S0, RATE, T30, SIGMA)
print(f"\n  ATM 30-day Call : ₹{c_atm:.4f}")
print(f"  ATM 30-day Put  : ₹{p_atm:.4f}")
print(f"  Forward price   : ₹{S0 * np.exp(RATE * T30):.4f}")


# ============================================================
# STEP 2 — ALL 5 GREEKS ANALYTICALLY
# ============================================================
hdr("STEP 2 — ALL 5 GREEKS")

def greeks(S, K, r, T, sigma):
    """Return dict of all 5 Greeks for call and put."""
    d1, d2 = bs_d1d2(S, K, r, T, sigma)
    nd1    = norm.pdf(d1)                   # N'(d1)
    Nd1    = norm.cdf(d1)                   # N(d1)
    Nd2    = norm.cdf(d2)

    delta_c =  Nd1
    delta_p =  Nd1 - 1

    gamma   = nd1 / (S * sigma * np.sqrt(T))   # same call & put

    # Theta: annualised → per calendar day (÷365)
    theta_c = (-(S * nd1 * sigma) / (2 * np.sqrt(T))
               - r * K * np.exp(-r * T) * Nd2) / 365
    theta_p = (-(S * nd1 * sigma) / (2 * np.sqrt(T))
               + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

    # Vega: per 1% absolute move in vol (not per unit)
    vega    = S * nd1 * np.sqrt(T) / 100

    # Rho: per 1% absolute move in rate
    rho_c   =  K * T * np.exp(-r * T) * Nd2  / 100
    rho_p   = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return {
        "d1": d1, "d2": d2,
        "delta_c": delta_c, "delta_p": delta_p,
        "gamma":   gamma,
        "theta_c": theta_c, "theta_p": theta_p,
        "vega":    vega,
        "rho_c":   rho_c,   "rho_p":   rho_p,
    }

g_atm = greeks(S0, S0, RATE, T30, SIGMA)
print(f"\n  ATM 30-day option  (S={S0:.0f}, K={S0:.0f}, σ={SIGMA*100:.0f}%, r={RATE*100:.1f}%)\n")
print(f"  {'Greek':<14}  {'Call':>14}  {'Put':>14}  {'Units / Interpretation'}")
print(f"  {'-'*72}")
rows = [
    ("Delta",  g_atm['delta_c'], g_atm['delta_p'], "₹ change per ₹1 spot move"),
    ("Gamma",  g_atm['gamma'],   g_atm['gamma'],   "Delta change per ₹1 spot move"),
    ("Theta",  g_atm['theta_c'], g_atm['theta_p'], "₹ per calendar day (time decay)"),
    ("Vega",   g_atm['vega'],    g_atm['vega'],    "₹ per 1% absolute vol move"),
    ("Rho",    g_atm['rho_c'],   g_atm['rho_p'],   "₹ per 1% absolute rate move"),
]
for nm, vc, vp, note in rows:
    print(f"  {nm:<14}  {vc:>14.6f}  {vp:>14.6f}  {note}")


# ============================================================
# STEP 3 — GREEKS BEHAVIOUR ANALYSIS
# ============================================================
hdr("STEP 3 — GREEKS BEHAVIOUR ANALYSIS  (ATM 30-day option)")

# (a) Spot sensitivity: spot from 21000 to 26000
print("\n  (a) GREEKS vs SPOT  (K=23500, T=30d, σ=16%)")
spots_a = np.arange(21000, 26500, 500)
print(f"  {'Spot':>7}  {'CallΔ':>7}  {'PutΔ':>7}  {'Gamma':>9}  {'Theta_c':>9}  {'Vega':>9}  {'Rho_c':>8}")
print(f"  {'-'*67}")
for sp in spots_a:
    g = greeks(sp, S0, RATE, T30, SIGMA)
    print(f"  {sp:>7,.0f}  {g['delta_c']:>7.4f}  {g['delta_p']:>7.4f}  "
          f"{g['gamma']:>9.6f}  {g['theta_c']:>9.4f}  {g['vega']:>9.4f}  {g['rho_c']:>8.4f}")

# (b) Time decay: 90 → 1 day
print(f"\n  (b) GREEKS vs TIME TO EXPIRY  (S=K=23500, σ=16%)")
print(f"  {'Days':>6}  {'CallΔ':>7}  {'PutΔ':>7}  {'Gamma':>9}  {'Theta_c':>9}  {'Vega':>9}")
print(f"  {'-'*57}")
for td in [90, 60, 45, 30, 21, 14, 7, 3, 1]:
    T = td / 365
    g = greeks(S0, S0, RATE, T, SIGMA)
    print(f"  {td:>6}  {g['delta_c']:>7.4f}  {g['delta_p']:>7.4f}  "
          f"{g['gamma']:>9.6f}  {g['theta_c']:>9.4f}  {g['vega']:>9.4f}")

# (c) Vol sensitivity: 10% to 30%
print(f"\n  (c) GREEKS vs IMPLIED VOL  (S=K=23500, T=30d)")
print(f"  {'Vol%':>6}  {'CallΔ':>7}  {'PutΔ':>7}  {'Gamma':>9}  {'Theta_c':>9}  {'Vega':>9}")
print(f"  {'-'*57}")
for vol_pct in [10, 12, 14, 16, 18, 20, 25, 30]:
    g = greeks(S0, S0, RATE, T30, vol_pct / 100)
    print(f"  {vol_pct:>5}%  {g['delta_c']:>7.4f}  {g['delta_p']:>7.4f}  "
          f"{g['gamma']:>9.6f}  {g['theta_c']:>9.4f}  {g['vega']:>9.4f}")


# ============================================================
# STEP 4 — MONEYNESS ANALYSIS (30-day expiry)
# ============================================================
hdr("STEP 4 — MONEYNESS ANALYSIS  (T=30 days, σ=16%)")

def moneyness_label(K, S):
    r_ = K / S
    if   r_ < 0.92:  return "DOTM-P"
    elif r_ < 0.97:  return " ITM-P"
    elif r_ < 1.03:  return "   ATM"
    elif r_ < 1.08:  return " ITM-C"
    else:             return "DOTM-C"

print(f"\n  {'Strike':>7}  {'Money':>7}  {'Call ₹':>9}  {'Put ₹':>9}  {'C.Δ':>7}  {'P.Δ':>7}  "
      f"{'Gamma':>9}  {'Theta_c':>9}  {'Vega':>8}  {'Rho_c':>8}")
print(f"  {'-'*97}")

csv_rows = []
for K in STRIKES:
    lbl  = moneyness_label(K, S0)
    c    = bs_call(S0, K, RATE, T30, SIGMA)
    p    = bs_put(S0, K, RATE, T30, SIGMA)
    g    = greeks(S0, K, RATE, T30, SIGMA)
    lm   = np.log(K / S0)
    print(f"  {K:>7.0f}  {lbl:>7}  {c:>9.2f}  {p:>9.2f}  {g['delta_c']:>7.4f}  {g['delta_p']:>7.4f}  "
          f"{g['gamma']:>9.6f}  {g['theta_c']:>9.4f}  {g['vega']:>8.4f}  {g['rho_c']:>8.4f}")
    csv_rows.append({
        "strike": K, "moneyness": lbl.strip(),
        "log_moneyness": round(lm, 5),
        "call_price": round(c, 4), "put_price": round(p, 4),
        "delta_call": round(g['delta_c'], 6), "delta_put": round(g['delta_p'], 6),
        "gamma": round(g['gamma'], 8),
        "theta_call": round(g['theta_c'], 6), "theta_put": round(g['theta_p'], 6),
        "vega": round(g['vega'], 6),
        "rho_call": round(g['rho_c'], 6), "rho_put": round(g['rho_p'], 6),
    })

# Key insights
atm_i   = list(STRIKES).index(S0)
ditm_k  = STRIKES[0]
dotm_k  = STRIKES[-1]
g_ditm  = greeks(S0, ditm_k, RATE, T30, SIGMA)
g_dotm  = greeks(S0, dotm_k, RATE, T30, SIGMA)
print(f"\n  INSIGHTS:")
print(f"  • Delta DITM call ({ditm_k}): {g_ditm['delta_c']:.4f} ≈ 1 → behaves like owning the stock")
print(f"  • Delta DOTM call ({dotm_k}): {g_dotm['delta_c']:.4f} ≈ 0 → lottery ticket")
print(f"  • Gamma is highest at ATM ({S0}): {g_atm['gamma']:.6f} — Delta changes most rapidly here")
print(f"  • Vega highest ATM: ₹{g_atm['vega']:.4f} per 1% vol vs DOTM: ₹{g_dotm['vega']:.4f}")


# ============================================================
# STEP 5 — PRACTICAL INTERVIEW QUESTIONS
# ============================================================
hdr("STEP 5 — INTERVIEW QUESTIONS")

# Q1: Delta hedge approximation
delta_q1   = 0.60
spot_move  = 100
price_chg  = delta_q1 * spot_move
print(f"\n  Q1: If Delta = {delta_q1} and spot rises ₹{spot_move:,.0f}")
print(f"  → Option price changes by ≈ ₹{price_chg:.2f}")
print(f"     (Delta × ΔS = {delta_q1} × {spot_move})")
print(f"     With Gamma correction: ΔP ≈ Δ·ΔS + ½·Γ·ΔS² "
      f"= {delta_q1*spot_move:.2f} + {0.5*g_atm['gamma']*spot_move**2:.2f} "
      f"= ₹{delta_q1*spot_move + 0.5*g_atm['gamma']*spot_move**2:.2f}")

# Q2: Gamma for large moves
print(f"\n  Q2: Delta-neutral but worried about large moves — GAMMA matters.")
print(f"  ATM 30d Gamma = {g_atm['gamma']:.6f}")
print(f"  → For a ₹500 spot move: Delta changes by {g_atm['gamma']*500:.4f}")
print(f"    (Gamma = {g_atm['gamma']:.6f} means Delta shifts {g_atm['gamma']:.4f} per ₹1 move)")
print(f"    Long Gamma = you profit from large moves in either direction (convexity).")

# Q3: Theta
print(f"\n  Q3: Theta for ATM 30-day call = ₹{g_atm['theta_c']:.4f} per calendar day")
print(f"  → The option loses ₹{abs(g_atm['theta_c']):.2f} per day purely from time decay.")
print(f"     Over a weekend (2 days): -₹{abs(g_atm['theta_c'])*2:.2f}")
print(f"     Over 1 week (7 days)   : -₹{abs(g_atm['theta_c'])*7:.2f}")

# Q4: Vega vs time to expiry
print(f"\n  Q4: Vega as option approaches expiry (S=K=23500, σ=16%)")
vega_series = [(td, greeks(S0, S0, RATE, td/365, SIGMA)['vega']) for td in [90,60,30,14,7,3,1]]
for td, v in vega_series:
    bar = "█" * int(v / 0.5)
    print(f"     {td:>3}d: Vega = ₹{v:>6.4f}  {bar}")
print(f"  → Vega DECAYS as expiry approaches (proportional to √T).")
print(f"     90d Vega = ₹{vega_series[0][1]:.2f}  vs  1d Vega = ₹{vega_series[-1][1]:.4f} "
      f"({vega_series[0][1]/vega_series[-1][1]:.0f}× larger at 90d)")


# ============================================================
# SAVE CSV
# ============================================================
df_csv = pd.DataFrame(csv_rows)
df_csv.to_csv("greeks_full_table.csv", index=False)
print(f"\n  greeks_full_table.csv saved.")


# ============================================================
# STEP 6 — PLOTS
# ============================================================
SPOTS_PLOT = np.linspace(19000, 28000, 400)
COLORS5    = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd"]
STRIKES5   = [21000, 22000, 23500, 24500, 26000]

# ── Plot 1: Call & Put price vs spot ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, flag, title in zip(axes, ["call","put"], ["Call Price","Put Price"]):
    for j, K in enumerate(STRIKES5):
        prices_v = np.array([bs_call(sp, K, RATE, T30, SIGMA) if flag=="call"
                             else bs_put(sp, K, RATE, T30, SIGMA)
                             for sp in SPOTS_PLOT])
        ax.plot(SPOTS_PLOT, prices_v, lw=2, color=COLORS5[j], label=f"K={K:,}")
    ax.axvline(S0, color="grey", lw=1.2, ls="--", alpha=0.7, label="Spot=23500")
    ax.set_xlabel("Spot Price (₹)", fontsize=11)
    ax.set_ylabel("Option Price (₹)", fontsize=11)
    ax.set_title(f"BS {title} vs Spot  (T=30d, σ=16%)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))

plt.suptitle("Black-Scholes Option Prices vs Spot — Nifty50",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("bs_price_vs_spot.png", dpi=150, bbox_inches="tight")
plt.close()
print("  bs_price_vs_spot.png saved.")

# ── Plot 2: All 5 Greeks vs Spot ─────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes_flat  = axes.flatten()

greek_specs = [
    ("Delta (Call)",  lambda sp: greeks(sp, S0, RATE, T30, SIGMA)['delta_c'], "#1f77b4"),
    ("Delta (Put)",   lambda sp: greeks(sp, S0, RATE, T30, SIGMA)['delta_p'], "#ff7f0e"),
    ("Gamma",         lambda sp: greeks(sp, S0, RATE, T30, SIGMA)['gamma'],   "#2ca02c"),
    ("Theta (Call/day)", lambda sp: greeks(sp, S0, RATE, T30, SIGMA)['theta_c'], "#d62728"),
    ("Vega (per 1%vol)", lambda sp: greeks(sp, S0, RATE, T30, SIGMA)['vega'],  "#9467bd"),
    ("Rho (Call/1%r)",   lambda sp: greeks(sp, S0, RATE, T30, SIGMA)['rho_c'], "#8c564b"),
]
for i, (name, fn, col) in enumerate(greek_specs):
    ax = axes_flat[i]
    vals = np.array([fn(sp) for sp in SPOTS_PLOT])
    ax.plot(SPOTS_PLOT, vals, lw=2.2, color=col)
    ax.axvline(S0, color="grey", lw=1, ls="--", alpha=0.6)
    ax.axhline(0, color="black", lw=0.7, alpha=0.4)
    ax.set_xlabel("Spot (₹)", fontsize=10)
    ax.set_ylabel(name, fontsize=10)
    ax.set_title(name, fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

axes_flat[5].set_visible(False)   # hide 6th subplot since we have 5 Greeks + Delta-put = 6 shown

plt.suptitle("All 5 Black-Scholes Greeks vs Spot  (K=23500, T=30d, σ=16%)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("greeks_vs_spot.png", dpi=150, bbox_inches="tight")
plt.close()
print("  greeks_vs_spot.png saved.")

# ── Plot 3: Theta Decay Curve ─────────────────────────────
days_arr = np.linspace(90, 0.5, 500)
T_arr    = days_arr / 365

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: call price vs days-to-expiry for 3 strikes
for j, K in enumerate([22000, 23500, 25000]):
    prices_t = np.array([bs_call(S0, K, RATE, t, SIGMA) for t in T_arr])
    axes[0].plot(days_arr, prices_t, lw=2.2, color=COLORS5[j],
                 label=f"K={K:,}  ({'ITM' if K<S0 else 'ATM' if K==S0 else 'OTM'})")
axes[0].set_xlabel("Days to Expiry", fontsize=11)
axes[0].set_ylabel("Call Price (₹)", fontsize=11)
axes[0].set_title("Theta Decay — Call Price vs Time", fontsize=12, fontweight="bold")
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)
axes[0].invert_xaxis()

# Right: annualised Theta rate vs days
theta_atm = np.array([greeks(S0, S0, RATE, t, SIGMA)['theta_c'] for t in T_arr])
axes[1].plot(days_arr, theta_atm, lw=2.2, color="#d62728")
axes[1].fill_between(days_arr, theta_atm, 0, alpha=0.2, color="#d62728")
axes[1].axvline(30, color="grey", lw=1.2, ls="--", label="30 days")
axes[1].axvline(7,  color="orange", lw=1.2, ls="--", label="7 days")
axes[1].set_xlabel("Days to Expiry", fontsize=11)
axes[1].set_ylabel("Theta (₹/day)", fontsize=11)
axes[1].set_title("ATM Call Theta vs Time to Expiry", fontsize=12, fontweight="bold")
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)
axes[1].invert_xaxis()

plt.suptitle("Theta Decay Analysis — Nifty50 ATM Options",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("theta_decay.png", dpi=150, bbox_inches="tight")
plt.close()
print("  theta_decay.png saved.")

# ── Plot 4: Vega Surface (spot × maturity heatmap) ────────
spots_heat = np.linspace(20000, 27000, 60)
days_heat  = np.array([1, 3, 7, 14, 21, 30, 45, 60, 90])
vega_mat   = np.zeros((len(days_heat), len(spots_heat)))

for i, td in enumerate(days_heat):
    for j, sp in enumerate(spots_heat):
        vega_mat[i, j] = greeks(sp, S0, RATE, td/365, SIGMA)['vega']

fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(
    vega_mat,
    aspect = "auto",
    origin = "lower",
    cmap   = "plasma",
    extent = [spots_heat[0], spots_heat[-1], 0, len(days_heat)-1],
)
ax.set_yticks(range(len(days_heat)))
ax.set_yticklabels([f"{d}d" for d in days_heat])
ax.set_xlabel("Spot Price (₹)", fontsize=11)
ax.set_ylabel("Days to Expiry", fontsize=11)
ax.set_title("Vega Surface — Option Vega (₹ per 1% vol move)",
             fontsize=13, fontweight="bold", pad=12)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
ax.axvline(S0, color="white", lw=1.8, ls="--", alpha=0.9, label=f"ATM ₹{S0:,}")
cb = plt.colorbar(im, ax=ax, label="Vega (₹ per 1% vol move)")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("vega_surface.png", dpi=150, bbox_inches="tight")
plt.close()
print("  vega_surface.png saved.")

print("\n" + "=" * 68)
print("  ALL DONE.")
print("=" * 68)

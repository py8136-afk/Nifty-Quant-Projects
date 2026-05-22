"""
=============================================================
  GARCH-VaR + Expected Shortfall Engine — Indian Indices
  Author  : Prateek (IIM Jammu, MBA 2025-27)
  Methods : Historical Simulation · Parametric · GARCH(1,1)
            Expected Shortfall · Kupiec POF Backtest
  Indices : Nifty 50 · Bank Nifty · IT · Pharma · Auto
=============================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import yfinance as yf
warnings.filterwarnings("ignore")

try:
    from arch import arch_model
    ARCH_OK = True
except ImportError:
    ARCH_OK = False
    print("⚠  arch not found — GARCH columns will be N/A")
    print("   pip install arch")

# ── CONFIG ────────────────────────────────────────────────────────────────────
SYMBOLS = {
    "^NSEI":      "Nifty 50",
    "^NSEBANK":   "Bank Nifty",
    "^CNXIT":     "Nifty IT",
    "^CNXPHARMA": "Nifty Pharma",
    "^CNXAUTO":   "Nifty Auto",
}
PERIOD      = "5y"
CONF_LEVELS = [0.95, 0.99]
OUT_DIR     = os.path.dirname(os.path.abspath(__file__))

# Dark theme palette
BG      = "#0d1117"
PANEL   = "#161b22"
TEXT    = "#e6edf3"
SUBTEXT = "#8b949e"
C       = ["#4fc3f7", "#ef5350", "#66bb6a", "#ffa726", "#ab47bc", "#1565c0",
           "#2e7d32", "#b71c1c"]


# ══════════════════════════════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════════════════════════════

def download_returns(symbols, period):
    print(f"\n📥 Downloading {period} daily data …")
    data = {}
    for ticker, name in symbols.items():
        try:
            raw = yf.download(ticker, period=period, progress=False,
                              auto_adjust=True)
            if raw.empty:
                print(f"   ⚠  {name}: no data returned")
                continue
            close = raw["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.squeeze()
            close = close.dropna()
            rets  = np.log(close / close.shift(1)).dropna()
            data[ticker] = {"name": name, "close": close, "returns": rets}
            print(f"   ✓  {name:<18}  {len(rets):,} trading days  "
                  f"({str(rets.index[0].date())} → {str(rets.index[-1].date())})")
        except Exception as e:
            print(f"   ✗  {name}: {e}")
    return data


# ══════════════════════════════════════════════════════════════════════════════
#  VAR METHODS
# ══════════════════════════════════════════════════════════════════════════════

def hist_var(returns, conf):
    """Historical Simulation — sort returns, find tail percentile. No dist assumption."""
    return float(-np.percentile(returns, (1 - conf) * 100))


def param_var(returns, conf):
    """Parametric — normal distribution, use sample mean + std."""
    mu    = float(returns.mean())
    sigma = float(returns.std())
    z     = stats.norm.ppf(1 - conf)
    return float(-(mu + z * sigma))


def hist_es(returns, conf):
    """Historical Expected Shortfall — mean of all returns below VaR threshold."""
    threshold = np.percentile(returns, (1 - conf) * 100)
    tail      = returns[returns <= threshold]
    return float(-tail.mean()) if len(tail) > 0 else float("nan")


def garch_analysis(returns):
    """
    Fit GARCH(1,1) with constant mean. Returns:
      vars   : {0.95: var, 0.99: var}
      es     : {0.95: es,  0.99: es}
      cond_vol: pd.Series of daily conditional vol (decimal)
    """
    nan_result = ({c: float("nan") for c in CONF_LEVELS},
                  {c: float("nan") for c in CONF_LEVELS},
                  None)
    if not ARCH_OK:
        return nan_result

    try:
        ret_pct = returns.values * 100   # scale → better numerical conditioning
        model   = arch_model(ret_pct, mean="Constant", vol="Garch", p=1, q=1,
                             dist="normal")
        res     = model.fit(disp="off", show_warning=False)

        cond_vol = pd.Series(
            res.conditional_volatility / 100,
            index=returns.index,
        )
        # 1-day ahead variance forecast
        fc       = res.forecast(horizon=1, reindex=False)
        fcast_vol = float(np.sqrt(fc.variance.iloc[-1, 0]) / 100)
        mu_hat    = float(res.params["mu"] / 100)

        gvars, ges = {}, {}
        for conf in CONF_LEVELS:
            z      = stats.norm.ppf(1 - conf)      # negative, e.g. -1.645
            gvars[conf] = float(-(mu_hat + z * fcast_vol))
            phi_z  = stats.norm.pdf(-z)             # phi(|z|)
            ges[conf]   = float(-(mu_hat - fcast_vol * phi_z / (1 - conf)))

        return gvars, ges, cond_vol

    except Exception as e:
        print(f"\n     ⚠  GARCH fit failed: {e}")
        return nan_result


# ══════════════════════════════════════════════════════════════════════════════
#  KUPIEC POF BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

def kupiec_pof(returns, var_99):
    """
    Kupiec (1995) Proportion of Failures test on Historical 99% VaR.

    H0 : breach rate = p = 0.01
    LR = -2 * ln[(1-p)^(T-N) * p^N  /  (1-N/T)^(T-N) * (N/T)^N]
    LR ~ chi²(1) under H0   |   critical value at 95% conf = 3.841
    """
    p = 0.01
    T = len(returns)
    N = int(np.sum(returns < -var_99))

    if N == 0:
        return dict(N=0, T=T, obs_pct=0.0, exp_pct=1.0,
                    LR=float("nan"), pval=float("nan"), verdict="PASS (0 breaches)")
    if N == T:
        return dict(N=N, T=T, obs_pct=100.0, exp_pct=1.0,
                    LR=float("nan"), pval=float("nan"), verdict="FAIL (all breach)")

    p_hat = N / T
    lr = -2 * (
        (T - N) * np.log(1 - p) + N * np.log(p)
        - (T - N) * np.log(1 - p_hat) - N * np.log(p_hat)
    )
    pval    = float(1 - stats.chi2.cdf(lr, df=1))
    chi2_cv = stats.chi2.ppf(0.95, df=1)   # 3.841

    return dict(
        N=N, T=T,
        obs_pct=round(p_hat * 100, 3),
        exp_pct=round(p * 100, 3),
        LR=round(lr, 4),
        pval=round(pval, 4),
        cv=round(chi2_cv, 3),
        verdict="PASS ✓" if lr < chi2_cv else "FAIL ✗",
    )


# ══════════════════════════════════════════════════════════════════════════════
#  ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════

def run_all(data):
    print("\n📐 Running models …")
    results   = {}
    nifty_aux = {}   # GARCH internals for Nifty 50 plots

    for ticker, info in data.items():
        name = info["name"]
        rets = info["returns"]
        sys.stdout.write(f"   {name:<18} … ")
        sys.stdout.flush()

        h95 = hist_var(rets, 0.95)
        h99 = hist_var(rets, 0.99)
        p95 = param_var(rets, 0.95)
        p99 = param_var(rets, 0.99)
        e95 = hist_es(rets, 0.95)
        e99 = hist_es(rets, 0.99)

        gvars, ges, cond_vol = garch_analysis(rets)
        kup = kupiec_pof(rets, h99)

        results[ticker] = {
            "name":       name,
            "n":          len(rets),
            "hist95":     h95,  "hist99":    h99,
            "param95":    p95,  "param99":   p99,
            "garch95":    gvars[0.95], "garch99": gvars[0.99],
            "es95":       e95,  "es99":      e99,
            "garch_es95": ges[0.95],   "garch_es99": ges[0.99],
            "kupiec":     kup,
        }

        if ticker == "^NSEI":
            nifty_aux = {"cond_vol": cond_vol, "returns": rets}

        print("✓")

    return results, nifty_aux


# ══════════════════════════════════════════════════════════════════════════════
#  OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(results):
    W = 120
    print("\n" + "=" * W)
    print("  GARCH-VaR + Expected Shortfall  —  Indian Indices (5-Year Daily)")
    print("  Values = daily % loss at stated confidence  |  positive number = loss")
    print("=" * W)

    hdr = (f"  {'Index':<16}  "
           f"{'Hist VaR':^15}  "
           f"{'Param VaR':^15}  "
           f"{'GARCH VaR':^15}  "
           f"{'Hist ES':^15}  "
           f"Kupiec 99%")
    sub = (f"  {'':16}  "
           f"{'95%':^7}{'99%':^8}  "
           f"{'95%':^7}{'99%':^8}  "
           f"{'95%':^7}{'99%':^8}  "
           f"{'95%':^7}{'99%':^8}  "
           f"Result   LR     p-val  Breaches")
    print(hdr)
    print(sub)
    print("  " + "─" * (W - 2))

    for ticker, r in results.items():
        k = r["kupiec"]
        g95 = f"{r['garch95']*100:5.2f}%" if not np.isnan(r["garch95"]) else "  N/A "
        g99 = f"{r['garch99']*100:5.2f}%" if not np.isnan(r["garch99"]) else "  N/A "
        lr_str   = f"{k['LR']:.3f}"  if not np.isnan(k.get("LR",  float("nan"))) else "  N/A"
        pval_str = f"{k['pval']:.4f}" if not np.isnan(k.get("pval",float("nan"))) else "  N/A"
        print(
            f"  {r['name']:<16}  "
            f"{r['hist95']*100:5.2f}%  {r['hist99']*100:5.2f}%   "
            f"{r['param95']*100:5.2f}%  {r['param99']*100:5.2f}%   "
            f"{g95}  {g99}   "
            f"{r['es95']*100:5.2f}%  {r['es99']*100:5.2f}%   "
            f"{k['verdict']:<8} {lr_str:>7} {pval_str:>7}  "
            f"{k['N']}/{k['T']}"
        )

    print("  " + "─" * (W - 2))
    print()
    print("  Kupiec POF Test:")
    print("    H0 : breach rate = 1.0%  (correct 99% VaR)  |  "
          "chi²(1) critical = 3.841 at 95% confidence")
    print("    PASS → model well-calibrated   |   FAIL → mis-specified (over or under)")
    print("=" * W)


def save_csv(results):
    rows = []
    for ticker, r in results.items():
        k = r["kupiec"]
        rows.append({
            "ticker":           ticker,
            "name":             r["name"],
            "n_obs":            r["n"],
            "hist_var_95":      round(r["hist95"]  * 100, 4),
            "hist_var_99":      round(r["hist99"]  * 100, 4),
            "param_var_95":     round(r["param95"] * 100, 4),
            "param_var_99":     round(r["param99"] * 100, 4),
            "garch_var_95":     round(r["garch95"] * 100, 4) if not np.isnan(r["garch95"]) else "",
            "garch_var_99":     round(r["garch99"] * 100, 4) if not np.isnan(r["garch99"]) else "",
            "hist_es_95":       round(r["es95"]    * 100, 4),
            "hist_es_99":       round(r["es99"]    * 100, 4),
            "garch_es_95":      round(r["garch_es95"] * 100, 4) if not np.isnan(r["garch_es95"]) else "",
            "garch_es_99":      round(r["garch_es99"] * 100, 4) if not np.isnan(r["garch_es99"]) else "",
            "kupiec_N":         k["N"],
            "kupiec_T":         k["T"],
            "kupiec_obs_pct":   k["obs_pct"],
            "kupiec_exp_pct":   k["exp_pct"],
            "kupiec_LR":        k.get("LR", ""),
            "kupiec_pval":      k.get("pval", ""),
            "kupiec_verdict":   k["verdict"],
        })
    path = os.path.join(OUT_DIR, "garch_var_results.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"💾 CSV saved   → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def _style(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=SUBTEXT, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")


def plot_garch_vol(nifty_aux):
    """Plot 1: returns + GARCH conditional volatility for Nifty 50."""
    rets     = nifty_aux["returns"]
    cond_vol = nifty_aux.get("cond_vol")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    facecolor=BG, sharex=True)
    fig.suptitle("Nifty 50 — GARCH(1,1) Conditional Volatility  (5-Year)",
                 color=TEXT, fontsize=13, fontweight="bold", y=0.98)

    # Panel 1 — daily log returns
    colors_bar = [C[0] if r >= 0 else C[1] for r in rets.values]
    ax1.bar(rets.index, rets.values * 100, color=colors_bar, width=1.2, alpha=0.85)
    ax1.axhline(0, color=SUBTEXT, linewidth=0.5, linestyle="--")
    ax1.set_ylabel("Log Return (%)", color=SUBTEXT, fontsize=9)
    ax1.set_title("Daily Log Returns", color=TEXT, fontsize=10, pad=4)
    _style(ax1)

    # Panel 2 — conditional vol (annualised)
    if cond_vol is not None:
        ann = cond_vol * np.sqrt(252) * 100
        ax2.plot(ann.index, ann.values, color=C[2], linewidth=1.2,
                 label="GARCH Cond. Vol (annualised)")
        ax2.fill_between(ann.index, ann.values, alpha=0.18, color=C[2])
        ax2.axhline(ann.mean(), color=C[3], linewidth=1.2, linestyle="--",
                    label=f"Mean: {ann.mean():.1f}%")
        ax2.legend(facecolor="#1c2128", edgecolor="#30363d",
                   labelcolor=TEXT, fontsize=9)
    else:
        ax2.text(0.5, 0.5, "GARCH not available", transform=ax2.transAxes,
                 ha="center", color=SUBTEXT)

    ax2.set_ylabel("Annualised Vol (%)", color=SUBTEXT, fontsize=9)
    ax2.set_title("GARCH(1,1) Conditional Volatility", color=TEXT, fontsize=10, pad=4)
    ax2.set_xlabel("Date", color=SUBTEXT, fontsize=9)
    _style(ax2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUT_DIR, "nifty50_garch_vol.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"📊 Plot saved  → {path}")


def plot_var_comparison(nifty_result, nifty_aux):
    """Plot 2: Nifty 50 return distribution with all VaR / ES thresholds."""
    rets = nifty_aux["returns"].values * 100
    r    = nifty_result

    fig, ax = plt.subplots(figsize=(14, 7), facecolor=BG)
    _style(ax)

    # Histogram
    ax.hist(rets, bins=90, density=True, color="#264f78",
            edgecolor="none", alpha=0.72, label="Daily Returns (empirical)")

    # Normal fit overlay
    mu_fit, s_fit = rets.mean(), rets.std()
    xs = np.linspace(rets.min() - 0.5, rets.max() + 0.5, 400)
    ax.plot(xs, stats.norm.pdf(xs, mu_fit, s_fit),
            color=SUBTEXT, linewidth=1.5, linestyle="--",
            alpha=0.8, label="Normal fit")

    # VaR / ES vertical lines
    lines = [
        (r["hist95"],     C[0],          "Hist VaR 95%",   "--",  1.8),
        (r["hist99"],     "#1565c0",     "Hist VaR 99%",   "--",  2.2),
        (r["param95"],    C[2],          "Param VaR 95%",  "-.",  1.8),
        (r["param99"],    "#2e7d32",     "Param VaR 99%",  "-.",  2.2),
        (r["garch95"],    C[3],          "GARCH VaR 95%",  ":",   2.0),
        (r["garch99"],    "#e65100",     "GARCH VaR 99%",  ":",   2.4),
        (r["es95"],       C[1],          "ES 95%",          "-",   2.0),
        (r["es99"],       "#b71c1c",     "ES 99%",          "-",   2.4),
    ]
    for val, color, label, ls, lw in lines:
        if not np.isnan(val):
            ax.axvline(-val * 100, color=color, linewidth=lw, linestyle=ls,
                       label=f"{label}: {val*100:.2f}%", alpha=0.92)

    ax.set_xlabel("Daily Log Return (%)", color=SUBTEXT, fontsize=11)
    ax.set_ylabel("Density", color=SUBTEXT, fontsize=11)
    ax.set_title(
        "Nifty 50 — Return Distribution with VaR & ES Thresholds\n"
        "Historical Simulation  |  Parametric (Normal)  |  GARCH(1,1)"
        "  @  95% & 99% confidence",
        color=TEXT, fontsize=12, fontweight="bold", pad=10,
    )
    ax.tick_params(colors=SUBTEXT)
    ax.legend(facecolor="#1c2128", edgecolor="#30363d", labelcolor=TEXT,
              fontsize=8.5, ncol=2, loc="upper left")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "nifty50_var_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"📊 Plot saved  → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  📊  GARCH-VaR + Expected Shortfall Engine")
    print("  Indian Indices: Nifty 50 · Bank Nifty · IT · Pharma · Auto")
    print("=" * 65)

    data = download_returns(SYMBOLS, PERIOD)
    if not data:
        print("❌  No data. Check internet connection.")
        sys.exit(1)

    results, nifty_aux = run_all(data)

    print_summary(results)

    print("\n💾 Saving outputs …")
    save_csv(results)

    if "^NSEI" in results and nifty_aux:
        print("\n🎨 Generating Nifty 50 plots …")
        plot_garch_vol(nifty_aux)
        plot_var_comparison(results["^NSEI"], nifty_aux)

    print("\n" + "=" * 65)
    print("  ✅  COMPLETE")
    print(f"  📂  Output dir → {OUT_DIR}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()

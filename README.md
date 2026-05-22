# Nifty Quant Projects

Quantitative finance projects built on Indian markets — from risk modelling to factor research to derivatives pricing.

**Author:** Prateek Yadav | IIM Jammu MBA 2025–27 | CFA L2 Cleared  
**Stack:** Python · R · SQL · yfinance · arch · quantmod · scipy · sklearn · statsmodels

## Projects

| # | Project | Language | Key Result |
|---|---------|----------|------------|
| 1 | [GARCH-VaR + Expected Shortfall](./garch-var-expected-shortfall/) | Python | Kupiec-validated VaR on 5 Indian indices — all 5 pass at 95% confidence |
| 2 | [India Multi-Factor Model](./india-factor-model/) | R | 9.2% annualised return, IC 0.077, IC IR 0.25 on 19 NSE stocks |
| 3 | [BankNifty-Nifty Pairs Trade](./banknifty-nifty-pairs-trade/) | Python + SQL | ADF p=0.022 (stationary spread), rolling hedge ratio 2.41x, SQLite trade log |
| 4 | [Nifty Implied Vol Surface](./nifty-implied-vol-surface/) | Python | SVI-fitted surface, put skew +3.25%, inverted term structure 17.6%→14.9% |
| 5 | [Credit Risk Scoring](./credit-risk-scoring/) | R | XGBoost AUC 0.776, KS 0.463, EL=₹2,103 Cr (4.26% of portfolio) |
| 6 | [Portfolio Stress Testing](./portfolio-stress-testing/) | Python | COVID crash −36.2% (−₹1.81 Cr), HINDUNILVR most defensive (−18.4%) |
| 7 | [Monte Carlo Asian Options](./monte-carlo-asian-options/) | Python | Antithetic variates reduce variance by ~50%, Asian option priced with control variates |
| 8 | [PCA Risk Decomposition](./pca-risk-decomposition/) | Python | PC1 explains ~65% of portfolio variance (market beta), component VaR verified |
| 9 | [Black-Scholes Greeks](./black-scholes-greeks/) | Python | Full Greeks surface, put-call parity error <1e-10, Vega/Theta sensitivity tables |
| 10 | [Regime-Aware VaR](./regime-aware-var/) | Python | Original research — OLS slope regime classifier switches GARCH↔HistSim VaR, Kupiec-validated |

## What makes this different
- All projects use **Indian market data** (NSE/BSE) — not textbook US examples
- Projects 1–4 cover the core quant interview canon (VaR, factor models, pairs trading, vol surface)
- Projects 5–6 target banking/credit roles directly (IFRS 9 EL, RBI stress scenarios)
- Projects 7–9 target derivatives desk and quant developer interviews
- Project 10 is **original research** — regime-switching VaR built on a live algo trading classifier

## Interview questions covered
VaR vs Expected Shortfall · GARCH volatility clustering · Cointegration vs correlation · Hedge ratio construction · IV smile and skew · PD/LGD/EAD · Stress testing vs VaR · Antithetic variates · Marginal vs component VaR · Black-Scholes assumptions · Greeks behaviour · Regime switching models

## Related projects
- [MyAlgo](https://github.com/py8136-afk/MyAlgo) — Live automated Nifty options trading system (Iron Fly + Bull/Bear spreads, TOTP auto-login, launchd scheduling)
- [Nifty EMA Backtest](https://github.com/py8136-afk/Nifty_EMA_Backtest) — 5-year EMA crossover backtest, 42% return on ₹10L capital, Sharpe 1.53

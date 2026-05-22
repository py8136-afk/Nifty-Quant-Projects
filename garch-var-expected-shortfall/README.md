# Project 01 — GARCH-VaR + Expected Shortfall Engine
**Author:** Prateek Yadav | IIM Jammu MBA 2025–27 | CFA L2

## What this is
Four-method VaR framework on 5 Indian indices (Nifty 50, Bank Nifty, IT, Pharma, Auto).
Methods: Historical Simulation · Parametric · GARCH(1,1) conditional · Expected Shortfall
Validated with Kupiec POF backtest — all 5 indices pass at 95% confidence.

## Key findings
- Nifty IT riskiest: 3.91% hist VaR, 4.85% ES at 99%
- Parametric VaR understates risk vs Historical — fat tails in Indian markets
- GARCH adapts to volatility regimes; ES captures tail risk VaR misses
- All Kupiec tests pass: observed breach rate ~1.05% vs expected 1.0%

## Methods
| Method | Assumption | Use case |
|--------|-----------|----------|
| Historical Simulation | None | Benchmark |
| Parametric | Normal distribution | Fast estimate |
| GARCH(1,1) | Time-varying vol | Dynamic risk |
| Expected Shortfall | None | Tail risk beyond VaR |

## Stack
Python · yfinance · arch · scipy · matplotlib · pandas

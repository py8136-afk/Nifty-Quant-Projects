# ============================================================
#  India Multi-Factor Model  |  NSE 20-Stock Universe
#  Factors: Momentum, Reversal, Volatility
#  Period  : Jan 2020 – May 2025
# ============================================================

# ── 0. Install / Load packages ────────────────────────────────
pkgs <- c("quantmod", "PerformanceAnalytics", "dplyr", "ggplot2", "lubridate")
new  <- pkgs[!pkgs %in% installed.packages()[, "Package"]]
if (length(new)) install.packages(new, repos = "https://cloud.r-project.org", quiet = TRUE)

suppressPackageStartupMessages({
  library(quantmod)
  library(PerformanceAnalytics)
  library(dplyr)
  library(ggplot2)
  library(lubridate)
})

setwd("~/Desktop/Nifty_Quant_Projects/india-factor-model")

# ── 1. Universe ───────────────────────────────────────────────
tickers <- c(
  "INFY.NS", "TCS.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS",       # IT
  "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "SBIN.NS", # Banks
  "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS", # Pharma
  "MARUTI.NS", "TATAMOTORS.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "HEROMOTOCO.NS" # Auto
)

start_date <- "2019-01-01"   # extra year for warm-up
end_date   <- "2025-05-31"

# ── 2. Download price data ────────────────────────────────────
cat("Downloading price data from Yahoo Finance...\n")

price_list <- list()
for (tkr in tickers) {
  cat("  ->", tkr, "\n")
  tryCatch({
    raw <- getSymbols(tkr, src = "yahoo", from = start_date, to = end_date,
                      auto.assign = FALSE, warnings = FALSE)
    price_list[[tkr]] <- Ad(raw)          # adjusted close
  }, error = function(e) {
    cat("     [WARN] failed for", tkr, ":", conditionMessage(e), "\n")
  })
}

# Merge into one xts, forward-fill missing values
prices_xts <- do.call(merge, price_list)
colnames(prices_xts) <- names(price_list)
prices_xts <- na.locf(prices_xts, na.rm = FALSE)

# Keep only trading days where at least 15 stocks have prices
prices_xts <- prices_xts[rowSums(!is.na(prices_xts)) >= 15, ]

cat("Price matrix:", nrow(prices_xts), "days x", ncol(prices_xts), "stocks\n")

# ── 3. Compute monthly returns ────────────────────────────────
# Apply.monthly returns last-day-of-month close -> monthly log return
monthly_ret <- apply.monthly(prices_xts, function(x) {
  first_val <- as.numeric(x[1, ])
  last_val  <- as.numeric(x[nrow(x), ])
  ifelse(is.na(first_val) | first_val == 0, NA, (last_val - first_val) / first_val)
})

# Restrict to Jan 2020 onwards for factor construction
monthly_ret <- monthly_ret[index(monthly_ret) >= as.Date("2020-01-01"), ]

# Also need daily prices for vol factor
daily_prices <- prices_xts[index(prices_xts) >= as.Date("2019-06-01"), ]

# ── 4. Build factors month by month ───────────────────────────
month_ends <- index(monthly_ret)
n_months   <- length(month_ends)

factor_df <- data.frame()

for (i in seq_along(month_ends)) {
  today <- month_ends[i]

  # Need ~13 months of history + 60 days daily
  # Momentum: cumulative return months [-12, -1]  (skip last month)
  # Reversal : return of last 1 month (month -1)
  # Volatility: std dev of daily returns over last 60 trading days

  # Align monthly index
  if (i < 13) next   # need 12 months of prior monthly returns

  # Monthly returns for momentum window: months [i-12 .. i-1]
  mom_window   <- monthly_ret[(i - 12):(i - 1), ]   # 12 rows x stocks
  mom_12       <- apply(mom_window, 2, function(r) prod(1 + r, na.rm = TRUE) - 1)
  rev_1        <- as.numeric(monthly_ret[i - 1, ])   # 1-month return (month -1)
  names(rev_1) <- colnames(monthly_ret)

  # 60-day daily volatility ending at today
  daily_window <- daily_prices[index(daily_prices) <= today]
  daily_window <- tail(daily_window, 61)
  if (nrow(daily_window) < 30) next
  daily_rets_w <- diff(log(daily_window))
  vol_60       <- apply(daily_rets_w, 2, function(r) sd(r, na.rm = TRUE))
  inv_vol      <- ifelse(vol_60 == 0 | is.na(vol_60), NA, 1 / vol_60)

  # Gather into row
  stocks <- colnames(monthly_ret)
  row_df <- data.frame(
    month      = today,
    stock      = stocks,
    momentum   = as.numeric(mom_12[stocks]),
    reversal   = as.numeric(rev_1[stocks]),
    inv_vol    = as.numeric(inv_vol[stocks]),
    fwd_return = if (i < n_months) as.numeric(monthly_ret[i + 1, stocks]) else NA,
    stringsAsFactors = FALSE
  )
  factor_df <- rbind(factor_df, row_df)
}

cat("Factor rows built:", nrow(factor_df), "\n")

# ── 5. Cross-sectional ranking & composite score ──────────────
factor_df <- factor_df %>%
  group_by(month) %>%
  mutate(
    rank_mom   = rank(momentum,  na.last = "keep", ties.method = "average"),
    rank_rev   = rank(reversal,  na.last = "keep", ties.method = "average"),
    rank_ivol  = rank(inv_vol,   na.last = "keep", ties.method = "average"),
    composite  = (rank_mom + rank_ivol - rank_rev) / 3
  ) %>%
  ungroup()

# ── 6. Long-short portfolio ────────────────────────────────────
portfolio_returns <- factor_df %>%
  filter(!is.na(composite), !is.na(fwd_return)) %>%
  group_by(month) %>%
  filter(n() >= 10) %>%
  mutate(
    comp_rank = rank(composite, ties.method = "average"),
    n_stocks  = n(),
    position  = case_when(
      comp_rank >= n_stocks - 4 ~ 1 / 5,    # top 5 -> long
      comp_rank <= 5            ~ -1 / 5,   # bottom 5 -> short
      TRUE                      ~ 0
    )
  ) %>%
  summarise(
    port_return = sum(position * fwd_return, na.rm = TRUE),
    .groups = "drop"
  )

# ── 7. IC calculation ─────────────────────────────────────────
ic_monthly <- factor_df %>%
  filter(!is.na(composite), !is.na(fwd_return)) %>%
  group_by(month) %>%
  filter(n() >= 10) %>%
  summarise(
    ic = cor(composite, fwd_return, method = "spearman", use = "pairwise.complete.obs"),
    .groups = "drop"
  )

mean_ic <- mean(ic_monthly$ic, na.rm = TRUE)
sd_ic   <- sd(ic_monthly$ic, na.rm = TRUE)
ic_ir   <- mean_ic / sd_ic

# ── 8. Performance metrics ────────────────────────────────────
port_xts <- xts(portfolio_returns$port_return,
                order.by = as.Date(portfolio_returns$month))

ann_return  <- prod(1 + coredata(port_xts), na.rm = TRUE)^(12 / length(port_xts)) - 1
ann_vol     <- sd(coredata(port_xts), na.rm = TRUE) * sqrt(12)
sharpe      <- ann_return / ann_vol

# Max drawdown
cum_ret     <- cumprod(1 + coredata(port_xts))
running_max <- cummax(cum_ret)
drawdowns   <- cum_ret / running_max - 1
max_dd      <- min(drawdowns, na.rm = TRUE)

# ── 9. Print summary ──────────────────────────────────────────
cat("\n")
cat("=======================================================\n")
cat("       INDIA MULTI-FACTOR MODEL  —  SUMMARY\n")
cat("       Universe: 20 NSE stocks | Jan 2020 – May 2025\n")
cat("=======================================================\n")
cat(sprintf("  Annualised Return   : %+.2f%%\n", ann_return * 100))
cat(sprintf("  Annualised Volatility: %.2f%%\n",  ann_vol * 100))
cat(sprintf("  Sharpe Ratio        : %.3f\n",     sharpe))
cat(sprintf("  Max Drawdown        : %.2f%%\n",   max_dd * 100))
cat(sprintf("  Mean Monthly IC     : %.4f\n",     mean_ic))
cat(sprintf("  IC Standard Dev     : %.4f\n",     sd_ic))
cat(sprintf("  IC Information Ratio: %.3f\n",     ic_ir))
cat("=======================================================\n\n")

# Monthly breakdown table
monthly_summary <- portfolio_returns %>%
  mutate(
    year  = year(month),
    month_label = format(month, "%Y-%m"),
    cum_return  = cumprod(1 + port_return) - 1
  )

cat("Monthly Portfolio Returns (last 12):\n")
print(tail(monthly_summary %>% select(month_label, port_return, cum_return), 12),
      row.names = FALSE)

# ── 10. Save CSV ───────────────────────────────────────────────
out_df <- factor_df %>%
  select(month, stock, momentum, reversal, inv_vol, composite, fwd_return) %>%
  left_join(portfolio_returns, by = "month") %>%
  left_join(ic_monthly,        by = "month")

write.csv(out_df, "factor_performance.csv", row.names = FALSE)
cat("\nSaved: factor_performance.csv\n")

# ── 11. Plot 1: Cumulative Return ─────────────────────────────
cum_df <- monthly_summary %>%
  select(month, port_return) %>%
  mutate(
    cum_idx = cumprod(1 + port_return),
    month   = as.Date(month)
  )

p1 <- ggplot(cum_df, aes(x = month, y = cum_idx)) +
  geom_line(colour = "#1f77b4", linewidth = 1.1) +
  geom_hline(yintercept = 1, linetype = "dashed", colour = "grey60") +
  geom_ribbon(aes(ymin = 1, ymax = cum_idx),
              fill = "#1f77b4", alpha = 0.12) +
  scale_x_date(date_breaks = "6 months", date_labels = "%b %Y") +
  scale_y_continuous(labels = scales::number_format(accuracy = 0.01)) +
  labs(
    title    = "India Multi-Factor Model — Cumulative Return",
    subtitle = "Long-short portfolio | 20 NSE stocks | Monthly rebalance",
    x = NULL, y = "Growth of ₹1"
  ) +
  theme_minimal(base_size = 13) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title  = element_text(face = "bold"))

ggsave("factor_cumulative_return.png", p1, width = 10, height = 5, dpi = 150)
cat("Saved: factor_cumulative_return.png\n")

# ── 12. Plot 2: Monthly IC Bar Chart ──────────────────────────
ic_plot_df <- ic_monthly %>%
  mutate(
    month = as.Date(month),
    fill  = ifelse(ic >= 0, "Positive", "Negative")
  )

p2 <- ggplot(ic_plot_df, aes(x = month, y = ic, fill = fill)) +
  geom_col(width = 20) +
  geom_hline(yintercept = 0,       colour = "grey30",  linewidth = 0.5) +
  geom_hline(yintercept = mean_ic, colour = "#e15759", linewidth = 0.8,
             linetype = "dashed") +
  annotate("text", x = min(ic_plot_df$month),
           y = mean_ic + 0.03, label = sprintf("Mean IC = %.3f", mean_ic),
           colour = "#e15759", hjust = 0, size = 3.5) +
  scale_fill_manual(values = c("Positive" = "#2ca02c", "Negative" = "#d62728"),
                    guide = "none") +
  scale_x_date(date_breaks = "6 months", date_labels = "%b %Y") +
  labs(
    title    = "Monthly IC — Composite Factor vs. Next-Month Return",
    subtitle = "Spearman rank correlation | India 20-stock universe",
    x = NULL, y = "Information Coefficient (IC)"
  ) +
  theme_minimal(base_size = 13) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title  = element_text(face = "bold"))

ggsave("monthly_ic.png", p2, width = 10, height = 5, dpi = 150)
cat("Saved: monthly_ic.png\n")

cat("\nAll done.\n")

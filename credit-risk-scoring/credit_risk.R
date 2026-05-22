# ============================================================
# Credit Risk Scoring — PD / LGD / EAD / Expected Loss
# Indian Corporate Loan Portfolio  (N = 2000)
# ============================================================

set.seed(42)

# ── Install missing packages ────────────────────────────────
pkgs <- c("randomForest", "xgboost", "pROC", "caret",
          "ggplot2", "dplyr", "gridExtra")
new  <- pkgs[!pkgs %in% installed.packages()[, "Package"]]
if (length(new)) install.packages(new, repos = "https://cloud.r-project.org",
                                  quiet = TRUE)

suppressPackageStartupMessages({
  library(randomForest)
  library(xgboost)
  library(pROC)
  library(caret)
  library(ggplot2)
  library(dplyr)
  library(gridExtra)
})

cat(rep("=", 60), "\n", sep = "")
cat("  CREDIT RISK SCORING MODEL — INDIAN CORPORATE LOANS\n")
cat(rep("=", 60), "\n\n", sep = "")

# ============================================================
# 1.  SIMULATE PORTFOLIO
# ============================================================
N <- 2000

industries <- c("Manufacturing", "IT", "Retail",
                "Real_Estate", "Infrastructure")

df <- data.frame(
  loan_amount      = round(runif(N, 1e6, 5e8)),   # ₹10L – ₹50Cr
  tenor_years      = sample(1:10, N, replace = TRUE),
  interest_rate    = round(runif(N, 8, 18), 2),
  debt_to_income   = round(runif(N, 0.1, 0.9), 3),
  credit_score     = round(runif(N, 300, 900)),
  collateral_ratio = round(runif(N, 0, 2), 3),
  industry         = sample(industries, N, replace = TRUE,
                            prob = c(0.30, 0.20, 0.20, 0.15, 0.15)),
  company_age_years= round(runif(N, 1, 50)),
  prev_defaults    = sample(0:3, N, replace = TRUE,
                            prob = c(0.65, 0.20, 0.10, 0.05))
)

# Logit to target ~12% default rate, correlated with risk drivers
logit <- -0.8 +
  1.8  * df$debt_to_income  +
  -0.004 * df$credit_score  +
  0.6  * df$prev_defaults   +
  -0.3 * df$collateral_ratio +
  0.05 * df$interest_rate   +
  -0.02 * df$company_age_years +
  ifelse(df$industry == "Real_Estate",    0.4, 0) +
  ifelse(df$industry == "Infrastructure",-0.3, 0)

prob_true <- 1 / (1 + exp(-logit))
df$default_flag <- rbinom(N, 1, prob_true)

cat(sprintf("  Portfolio size   : %d loans\n", N))
cat(sprintf("  Default rate     : %.1f%%\n", mean(df$default_flag) * 100))
cat(sprintf("  Total exposure   : ₹%.2f Cr\n\n",
            sum(df$loan_amount) / 1e7))

# ============================================================
# 2.  FEATURE PREP
# ============================================================
df$industry_f <- as.integer(factor(df$industry))   # numeric for XGB

features <- c("loan_amount", "tenor_years", "interest_rate",
              "debt_to_income", "credit_score", "collateral_ratio",
              "company_age_years", "prev_defaults", "industry_f")

X     <- df[, features]
y     <- df$default_flag
y_fac <- factor(y, labels = c("No", "Yes"))

# 70/30 split
idx_train <- createDataPartition(y_fac, p = 0.7, list = FALSE)
X_tr  <- X[idx_train, ];  y_tr  <- y[idx_train];  yf_tr <- y_fac[idx_train]
X_te  <- X[-idx_train,];  y_te  <- y[-idx_train];  yf_te <- y_fac[-idx_train]

# ============================================================
# 3.  LOGISTIC REGRESSION
# ============================================================
cat(rep("-", 60), "\n", sep = "")
cat("  MODEL 1 — LOGISTIC REGRESSION\n")
cat(rep("-", 60), "\n", sep = "")

lr_df_tr <- cbind(X_tr, default_flag = y_tr)
lr_df_te <- cbind(X_te, default_flag = y_te)

lr_model   <- glm(default_flag ~ ., data = lr_df_tr, family = binomial)
lr_prob_te <- predict(lr_model, newdata = X_te, type = "response")
lr_roc     <- roc(y_te, lr_prob_te, quiet = TRUE)
lr_auc     <- as.numeric(auc(lr_roc))

# KS statistic
ks_stat <- function(probs, labels) {
  o    <- order(probs, decreasing = TRUE)
  tp   <- cumsum(labels[o]) / sum(labels)
  fp   <- cumsum(1 - labels[o]) / sum(1 - labels)
  max(abs(tp - fp))
}
lr_ks <- ks_stat(lr_prob_te, y_te)
cat(sprintf("  AUC : %.4f   KS : %.4f\n\n", lr_auc, lr_ks))

# ============================================================
# 4.  RANDOM FOREST
# ============================================================
cat(rep("-", 60), "\n", sep = "")
cat("  MODEL 2 — RANDOM FOREST\n")
cat(rep("-", 60), "\n", sep = "")

rf_model   <- randomForest(x = X_tr, y = yf_tr,
                           ntree = 300, mtry = 3,
                           importance = TRUE)
rf_prob_te <- predict(rf_model, newdata = X_te, type = "prob")[, "Yes"]
rf_roc     <- roc(y_te, rf_prob_te, quiet = TRUE)
rf_auc     <- as.numeric(auc(rf_roc))
rf_ks      <- ks_stat(rf_prob_te, y_te)
cat(sprintf("  AUC : %.4f   KS : %.4f\n\n", rf_auc, rf_ks))

cat("  Variable Importance (top 5):\n")
imp <- importance(rf_model)[, "MeanDecreaseGini"]
imp_sorted <- sort(imp, decreasing = TRUE)[1:5]
for (nm in names(imp_sorted))
  cat(sprintf("    %-22s %.4f\n", nm, imp_sorted[nm]))
cat("\n")

# ============================================================
# 5.  XGBOOST
# ============================================================
cat(rep("-", 60), "\n", sep = "")
cat("  MODEL 3 — XGBOOST\n")
cat(rep("-", 60), "\n", sep = "")

dtrain <- xgb.DMatrix(data = as.matrix(X_tr), label = y_tr)
dtest  <- xgb.DMatrix(data = as.matrix(X_te), label = y_te)

xgb_params <- list(
  objective        = "binary:logistic",
  eval_metric      = "auc",
  eta              = 0.05,
  max_depth        = 4,
  subsample        = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 5
)

xgb_model <- xgb.train(
  params    = xgb_params,
  data      = dtrain,
  nrounds   = 200,
  watchlist = list(train = dtrain, test = dtest),
  verbose   = 0
)

xgb_prob_te <- predict(xgb_model, dtest)
xgb_roc     <- roc(y_te, xgb_prob_te, quiet = TRUE)
xgb_auc     <- as.numeric(auc(xgb_roc))
xgb_ks      <- ks_stat(xgb_prob_te, y_te)
cat(sprintf("  AUC : %.4f   KS : %.4f\n\n", xgb_auc, xgb_ks))

# ============================================================
# 6.  5-FOLD CROSS-VALIDATION (all models)
# ============================================================
cat(rep("-", 60), "\n", sep = "")
cat("  5-FOLD CROSS-VALIDATION\n")
cat(rep("-", 60), "\n", sep = "")

folds   <- createFolds(y_fac, k = 5, list = TRUE)
cv_aucs <- data.frame(LR = numeric(5), RF = numeric(5), XGB = numeric(5))

for (f in seq_along(folds)) {
  val_idx <- folds[[f]]
  tr_idx  <- setdiff(seq_len(N), val_idx)

  # LR
  lr_cv <- glm(default_flag ~ .,
               data   = cbind(X[tr_idx, ], default_flag = y[tr_idx]),
               family = binomial)
  p_lr  <- predict(lr_cv, newdata = X[val_idx, ], type = "response")
  cv_aucs$LR[f] <- as.numeric(auc(roc(y[val_idx], p_lr, quiet = TRUE)))

  # RF
  rf_cv <- randomForest(x = X[tr_idx, ], y = factor(y[tr_idx]),
                        ntree = 100)
  p_rf  <- predict(rf_cv, newdata = X[val_idx, ], type = "prob")[, "1"]
  cv_aucs$RF[f] <- as.numeric(auc(roc(y[val_idx], p_rf, quiet = TRUE)))

  # XGB
  dtr <- xgb.DMatrix(data = as.matrix(X[tr_idx, ]), label = y[tr_idx])
  dval<- xgb.DMatrix(data = as.matrix(X[val_idx, ]))
  xgb_cv <- xgb.train(params = xgb_params, data = dtr,
                      nrounds = 100, verbose = 0)
  p_xgb <- predict(xgb_cv, dval)
  cv_aucs$XGB[f]<- as.numeric(auc(roc(y[val_idx], p_xgb, quiet = TRUE)))

  cat(sprintf("  Fold %d — LR: %.4f  RF: %.4f  XGB: %.4f\n",
              f, cv_aucs$LR[f], cv_aucs$RF[f], cv_aucs$XGB[f]))
}

cv_mean <- colMeans(cv_aucs)
cv_sd   <- apply(cv_aucs, 2, sd)
cat(sprintf("\n  Mean   — LR: %.4f  RF: %.4f  XGB: %.4f\n",
            cv_mean["LR"], cv_mean["RF"], cv_mean["XGB"]))
cat(sprintf("  SD     — LR: %.4f  RF: %.4f  XGB: %.4f\n\n",
            cv_sd["LR"], cv_sd["RF"], cv_sd["XGB"]))

# ============================================================
# 7.  MODEL COMPARISON TABLE
# ============================================================
best_model_name <- names(which.max(c(LR = lr_auc, RF = rf_auc, XGB = xgb_auc)))

comparison <- data.frame(
  Model   = c("Logistic Regression", "Random Forest", "XGBoost"),
  AUC     = round(c(lr_auc,  rf_auc,  xgb_auc),  4),
  KS      = round(c(lr_ks,   rf_ks,   xgb_ks),   4),
  CV_AUC  = round(cv_mean,                        4),
  CV_SD   = round(cv_sd,                          4),
  Best    = c(ifelse(c("LR","RF","XGB") == best_model_name, "★", ""))
)

cat(rep("=", 60), "\n", sep = "")
cat("  MODEL COMPARISON TABLE\n")
cat(rep("=", 60), "\n", sep = "")
cat(sprintf("  %-22s %7s %7s %10s %8s\n",
            "Model", "AUC", "KS", "CV_AUC", "CV_SD"))
cat(rep("-", 60), "\n", sep = "")
for (i in seq_len(nrow(comparison))) {
  cat(sprintf("  %-22s %7.4f %7.4f %10.4f %8.4f  %s\n",
              comparison$Model[i], comparison$AUC[i], comparison$KS[i],
              comparison$CV_AUC[i], comparison$CV_SD[i], comparison$Best[i]))
}
cat(sprintf("\n  Best model: %s\n\n", best_model_name))

# ============================================================
# 8.  EXPECTED LOSS  (PD × LGD × EAD)
# ============================================================
cat(rep("=", 60), "\n", sep = "")
cat("  EXPECTED LOSS CALCULATION\n")
cat(rep("=", 60), "\n", sep = "")

# Use best model to score full portfolio
df_full      <- cbind(df, industry_f = as.integer(factor(df$industry)))
X_full       <- df_full[, features]

if (best_model_name == "LR") {
  df$PD <- predict(lr_model, newdata = X_full, type = "response")
} else if (best_model_name == "RF") {
  df$PD <- predict(rf_model, newdata = X_full, type = "prob")[, "Yes"]
} else {
  dfull_xgb <- xgb.DMatrix(data = as.matrix(X_full))
  df$PD     <- predict(xgb_model, dfull_xgb)
}

df$LGD <- pmin(pmax(1 - df$collateral_ratio, 0), 1)   # capped [0,1]
df$EAD <- df$loan_amount
df$EL  <- df$PD * df$LGD * df$EAD

total_ead <- sum(df$EAD)
total_el  <- sum(df$EL)

cat(sprintf("  Total Portfolio EAD : ₹%10.2f Cr\n", total_ead / 1e7))
cat(sprintf("  Total Expected Loss : ₹%10.2f Cr\n", total_el  / 1e7))
cat(sprintf("  EL as %% of Portfolio: %10.4f%%\n\n",  total_el / total_ead * 100))

# Industry breakdown
el_industry <- df %>%
  group_by(industry) %>%
  summarise(
    Loans      = n(),
    EAD_Cr     = round(sum(EAD)  / 1e7, 2),
    EL_Cr      = round(sum(EL)   / 1e7, 2),
    EL_pct     = round(sum(EL)   / sum(EAD) * 100, 3),
    Avg_PD     = round(mean(PD)  * 100, 2),
    Avg_LGD    = round(mean(LGD) * 100, 2),
    .groups    = "drop"
  ) %>%
  arrange(desc(EL_Cr))

cat(rep("=", 60), "\n", sep = "")
cat("  EL BY INDUSTRY\n")
cat(rep("=", 60), "\n", sep = "")
cat(sprintf("  %-16s %6s %10s %10s %8s %7s %7s\n",
            "Industry", "Loans", "EAD(₹Cr)", "EL(₹Cr)", "EL%", "AvgPD%","AvgLGD%"))
cat(rep("-", 70), "\n", sep = "")
for (i in seq_len(nrow(el_industry))) {
  r <- el_industry[i, ]
  cat(sprintf("  %-16s %6d %10.2f %10.2f %8.3f %7.2f %7.2f\n",
              r$industry, r$Loans, r$EAD_Cr, r$EL_Cr,
              r$EL_pct, r$Avg_PD, r$Avg_LGD))
}
cat(sprintf("  %-16s %6d %10.2f %10.2f %8.3f\n",
            "TOTAL", nrow(df), total_ead/1e7, total_el/1e7,
            total_el/total_ead*100))
cat("\n")

# ============================================================
# 9.  SAVE CSV
# ============================================================
write.csv(df[, c("loan_amount","tenor_years","interest_rate",
                 "debt_to_income","credit_score","collateral_ratio",
                 "industry","company_age_years","prev_defaults",
                 "default_flag","PD","LGD","EAD","EL")],
          "credit_scores.csv", row.names = FALSE)
cat("  credit_scores.csv saved.\n\n")

# ============================================================
# 10.  PLOTS
# ============================================================

# ── Plot 1: ROC curves ─────────────────────────────────────
png("model_comparison.png", width = 900, height = 700, res = 120)
par(mar = c(5, 5, 4, 2))
plot(lr_roc,  col = "#1f77b4", lwd = 2.5,
     main = "ROC Curves — Credit Risk Models",
     cex.main = 1.3, cex.axis = 1.1, cex.lab = 1.1)
plot(rf_roc,  col = "#2ca02c", lwd = 2.5, add = TRUE)
plot(xgb_roc, col = "#d62728", lwd = 2.5, add = TRUE)
abline(0, 1, lty = 2, col = "grey60")
legend("bottomright",
       legend = c(
         sprintf("Logistic Regression (AUC=%.4f)", lr_auc),
         sprintf("Random Forest       (AUC=%.4f)", rf_auc),
         sprintf("XGBoost             (AUC=%.4f)", xgb_auc)
       ),
       col = c("#1f77b4", "#2ca02c", "#d62728"),
       lwd = 2.5, bty = "n", cex = 1.0)
dev.off()
cat("  model_comparison.png saved.\n")

# ── Plot 2: EL by industry ────────────────────────────────
p <- ggplot(el_industry,
            aes(x = reorder(industry, -EL_Cr), y = EL_Cr,
                fill = industry)) +
  geom_col(width = 0.65, colour = "grey30", linewidth = 0.3) +
  geom_text(aes(label = sprintf("₹%.1fCr\n(%.2f%%)", EL_Cr, EL_pct)),
            vjust = -0.4, size = 3.4, fontface = "bold") +
  scale_fill_brewer(palette = "Set2") +
  labs(
    title    = "Expected Loss by Industry",
    subtitle = sprintf("Total Portfolio EL = ₹%.2f Cr  (%.3f%% of EAD)",
                       total_el/1e7, total_el/total_ead*100),
    x = "Industry", y = "Expected Loss (₹ Cr)"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none",
        plot.title    = element_text(face = "bold", size = 14),
        plot.subtitle = element_text(size = 10, colour = "grey40"),
        axis.text.x   = element_text(angle = 15, hjust = 1))

ggsave("el_by_industry.png", p, width = 9, height = 5.5, dpi = 150)
cat("  el_by_industry.png saved.\n\n")

cat(rep("=", 60), "\n", sep = "")
cat("  ALL DONE.\n")
cat(rep("=", 60), "\n", sep = "")

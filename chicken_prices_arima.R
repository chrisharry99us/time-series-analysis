# =============================================================================
# ARIMA Analysis of Monthly US Chicken Prices
# Dataset: chicken (astsa package in R)
# Period: January 2001 – December 2016 (192 monthly observations)
# Final Model: ARIMA(3,1,0)(0,0,1)[12]
# =============================================================================

library(astsa)
library(tseries)
library(forecast)

# =============================================================================
# 1. LOAD AND PLOT RAW DATA
# =============================================================================

# Figure 1: Raw series
plot(chicken,
     main = "Monthly Chicken Prices (cents per pound)",
     ylab = "Price (cents per pound)",
     xlab = "Year",
     col  = "steelblue",
     lwd  = 1.5)

# Initial observations:
# - Clear upward trend: ~65 cents/lb (2001) → ~115 cents/lb (2016)
# - No obvious seasonal pattern
# - Variance roughly constant → log transformation NOT needed
# - Upward trend → series is non-stationary; differencing required

# =============================================================================
# 2. STATIONARITY ASSESSMENT
# =============================================================================

# ADF test on raw series
adf.test(chicken)
# Result: DF = -2.396, p-value = 0.411 → FAIL to reject H0 → non-stationary

# Apply first difference to remove trend
chicken_diff <- diff(chicken)

# Figure 2: First-differenced series
plot(chicken_diff,
     main = "First Difference of Monthly Chicken Prices",
     ylab = "Change in Price (cents per pound)",
     xlab = "Year",
     col  = "steelblue",
     lwd  = 1.5)
abline(h = 0, lty = 2, col = "red")
# Fluctuates around zero with roughly constant variance → stationary

# ADF test on differenced series
adf.test(chicken_diff)
# Result: DF = -5.582, p-value < 0.01 → Reject H0 → stationary
# Conclusion: exactly ONE first difference required; no seasonal differencing

# =============================================================================
# 3. ACF AND PACF FOR MODEL IDENTIFICATION
# =============================================================================

# Figure 3: ACF and PACF of first-differenced series
par(mfrow = c(1, 2))
acf(chicken_diff,  lag.max = 40, main = "ACF — First-Differenced Chicken Prices")
pacf(chicken_diff, lag.max = 40, main = "PACF — First-Differenced Chicken Prices")
par(mfrow = c(1, 1))
# PACF: significant spikes at lags 1, 2 (and possibly 3) → AR(2) or AR(3) candidate
# ACF: slow decay → confirms AR structure
# Possible seasonal signal at lag 12 in ACF → consider seasonal MA(1)

# =============================================================================
# 4. MODEL SELECTION: AUTO.ARIMA
# =============================================================================
auto.arima(chicken, allowdrift = FALSE, trace = TRUE)
# Best model selected: ARIMA(3,1,0)(0,0,1)[12]
# AICc = 1.9612

# =============================================================================
# 5. FIT THREE CANDIDATE MODELS
# =============================================================================

# Model 1 (Final): ARIMA(3,1,0)(0,0,1)[12] — auto.arima selection
# AICc = 1.9612 | Best diagnostics
# Equation: (1 − 0.898B + 0.142B² + 0.126B³)(1−B)Xₜ = (1 + 0.290B¹²)Wₜ
#           Wₜ ~ WN(0, 0.389)
# Note: ar2 (p=0.161) and ar3 (p=0.094) not significant at 5% level
#       but dramatic diagnostic improvement justifies retention
fit1 <- sarima(chicken, p = 3, d = 1, q = 0, P = 0, D = 0, Q = 1, S = 12)

# Model 2: ARIMA(2,1,0)(0,0,1)[12]
# More parsimonious; worse Ljung-Box performance than Model 1
fit2 <- sarima(chicken, p = 2, d = 1, q = 0, P = 0, D = 0, Q = 1, S = 12)

# Model 3 (Baseline): ARIMA(2,1,0) — non-seasonal
# Severe Ljung-Box failure → eliminated
fit3 <- sarima(chicken, p = 2, d = 1, q = 0)

# =============================================================================
# 6. MODEL COMPARISON SUMMARY
# =============================================================================
# Model  | Spec                       | AICc   | Ljung-Box  | Selected
# -------|----------------------------|--------|------------|--------
# M1     | ARIMA(3,1,0)(0,0,1)[12]   | 1.9612 | Pass       | YES ✓
# M2     | ARIMA(2,1,0)(0,0,1)[12]   | higher | Acceptable | Runner-up
# M3     | ARIMA(2,1,0)              | —      | FAIL       | Eliminated
#
# Selection rationale: M1 achieves best AICc and cleanest diagnostics.
# The seasonal MA(1) term captures subtle month-to-month patterns at the
# annual lag, substantially improving fit over the non-seasonal baseline.

# =============================================================================
# 7. FINAL MODEL SUMMARY
# =============================================================================
cat("Final Model: ARIMA(3,1,0)(0,0,1)[12]\n")
cat("Equation: (1 − 0.8982B + 0.1416B² + 0.1255B³)(1−B)Xₜ = (1 + 0.2899B¹²)Wₜ\n")
cat("White Noise Variance: σ² = 0.3886\n")
cat("AICc = 1.9612\n\n")
cat("Key findings:\n")
cat("  - Chicken prices rose from ~65 to ~115 cents/lb (2001–2016)\n")
cat("  - First differencing achieves stationarity (ADF p < 0.01)\n")
cat("  - Subtle seasonal MA(1) term improves model fit at annual lag\n")
cat("  - AR(3) structure captures serial dependence in price changes\n")

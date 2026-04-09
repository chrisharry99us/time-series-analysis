# =============================================================================
# SARIMA Analysis of UK Quarterly Gas Consumption
# Dataset: UKgas (built-in R datasets package)
# Period: 1960 Q1 – 1986 Q4 (108 quarterly observations)
# Final Model: SARIMA(2,0,3)(1,1,0)[4] fitted to log-transformed series
# =============================================================================

library(forecast)
library(tseries)
library(astsa)

# =============================================================================
# 1. LOAD AND PLOT RAW DATA
# =============================================================================
data(UKgas)

# Figure 1: Raw series
plot(UKgas,
     main = "UK Quarterly Gas Consumption (1960–1986)",
     ylab = "Gas Consumption (MMTherms)",
     xlab = "Year")

# Initial observations:
# - Clear upward trend from 1960s through mid-1980s
# - Strong seasonality: peaks once per year (winter quarters)
# - Variance grows over time (heteroscedastic) → log transformation needed
# Decomposition model: Xt = st + mt + Yt

# =============================================================================
# 2. LOG TRANSFORMATION
# =============================================================================
logUKgas <- log(UKgas)

# Figure 2: Log-transformed series
plot(logUKgas,
     main = "Log of UK Quarterly Gas Consumption",
     ylab = "Log Gas Consumption",
     xlab = "Year")
# Variance now stable; seasonality and trend still visible

# =============================================================================
# 3. SEASONAL DIFFERENCING (lag = 4 for quarterly data)
# =============================================================================
logUKgas_sdiff <- diff(logUKgas, lag = 4)

# Figure 3: Seasonally differenced log series
plot(logUKgas_sdiff,
     main = "Seasonally Differenced Log UKgas — (1−B⁴)log(Xₜ)",
     ylab = "",
     xlab = "Year")
# Trend and seasonality removed; series appears stationary

# =============================================================================
# 4. STATIONARITY TEST: AUGMENTED DICKEY-FULLER
# =============================================================================
adf.test(logUKgas_sdiff)
# Result: DF = -3.8731, p-value = 0.0180
# Reject H0 (unit root) → series is stationary
# No further differencing required
# Working series: (1−B⁴)log(Xₜ) → SARIMA(p,0,q)(P,1,Q)[4] on log scale

# =============================================================================
# 5. ACF AND PACF FOR MODEL IDENTIFICATION
# =============================================================================
par(mfrow = c(1, 2))
acf(logUKgas_sdiff,  lag.max = 40, main = "ACF — Seasonally Differenced Log UKgas")
pacf(logUKgas_sdiff, lag.max = 40, main = "PACF — Seasonally Differenced Log UKgas")
par(mfrow = c(1, 1))
# ACF: uninformative — no clear MA cutoff
# PACF: significant spikes at lags 5 and 9 (possibly noise)
# → Rely on auto.arima and AICc comparison

# =============================================================================
# 6. MODEL SELECTION: AUTO.ARIMA
# =============================================================================

# Primary search — seasonal SARIMA on log series (explores 23 candidates)
auto.arima(logUKgas, seasonal = TRUE, trace = TRUE, ic = "aicc", allowdrift = FALSE)
# Best: SARIMA(2,0,3)(1,1,0)[4] — AICc = -168.15

# Comparison: auto.arima on pre-differenced data (non-seasonal)
auto.arima(logUKgas_sdiff, seasonal = FALSE, trace = TRUE, ic = "aicc", allowdrift = FALSE)
# Returns ARIMA(1,0,0) — AICc = -169.28 (simpler but misses seasonal structure)

# =============================================================================
# 7. FIT THREE CANDIDATE MODELS
# =============================================================================

# Model 1 (Final): SARIMA(2,0,3)(1,1,0)[4] — auto.arima best model
# AICc = -1.619 | Strongest diagnostics
# Equation: (1 − 0.218B − 0.764B²)(1 − 0.181B⁴)(1−B⁴)log(Xₜ)
#         = (1 − 0.321B − 0.807B² + 0.337B³)Wₜ
# Note: ar1 (p=0.105) and sar1 (p=0.101) not significant at 5% level
sarima(logUKgas, p = 2, d = 0, q = 3, P = 1, D = 1, Q = 0, S = 4)

# Model 2: SARIMA(2,0,3)(0,1,1)[4] — second best
# AICc = -168.84 | Significant seasonal MA coefficient
# Only model with significant seasonal coefficient
sarima(logUKgas, p = 2, d = 0, q = 3, P = 0, D = 1, Q = 1, S = 4)

# Model 3: SARIMA(2,0,2)(0,1,1)[4] — third best
# AICc = -167.37 | More parsimonious but weaker diagnostics
sarima(logUKgas, p = 2, d = 0, q = 2, P = 0, D = 1, Q = 1, S = 4)

# =============================================================================
# 8. MODEL COMPARISON SUMMARY
# =============================================================================
# Model  | Spec                      | AICc    | Ljung-Box | Selected
# -------|---------------------------|---------|-----------|--------
# M1     | SARIMA(2,0,3)(1,1,0)[4]  | -1.619  | All > 0.05| YES ✓
# M2     | SARIMA(2,0,3)(0,1,1)[4]  | -168.84 | All > 0.05| Runner-up
# M3     | SARIMA(2,0,2)(0,1,1)[4]  | -167.37 | Some fail | No
#
# Selection rationale: M1 achieves lowest AICc, clean residual ACF,
# all Ljung-Box p-values > 0.05. Insignificance of ar1 and sar1 is a
# weakness but overall evidence strongly favours M1.

# =============================================================================
# 9. FORECAST: 12-STEP AHEAD (3 YEARS: Q1 1987 – Q4 1989)
# =============================================================================
fit_final <- arima(logUKgas,
                   order    = c(2, 0, 3),
                   seasonal = list(order = c(1, 1, 0), period = 4))

plot(forecast(fit_final, h = 12),
     main = "12-Step Ahead Forecast — SARIMA(2,0,3)(1,1,0)[4] on Log Scale",
     xlab = "Year",
     ylab = "Log Gas Consumption")
# Forecast preserves seasonal pattern and upward trend
# 80% and 95% prediction intervals shown
# Note: forecast is on log scale

# Time Series Analysis

**Master of Data Science — Time Series**

This repository contains two time series analysis projects demonstrating the full ARIMA/SARIMA modeling workflow: exploratory analysis, transformation, stationarity testing, model identification, diagnostic checking, and forecasting — all in R.

---

## Projects

### 1. SARIMA Analysis of UK Quarterly Gas Consumption
**`uk_gas_sarima.R`**

Analysis of the built-in `UKgas` dataset covering quarterly UK gas consumption from 1960 Q1 to 1986 Q4 (108 observations).

**Workflow:**

| Step | Action | Result |
|------|--------|--------|
| 1 | Plot raw series | Upward trend + growing seasonal variance |
| 2 | Log transformation | Stabilizes variance |
| 3 | Lag-4 seasonal differencing | Removes trend and seasonality |
| 4 | Augmented Dickey-Fuller test | DF = −3.87, p = 0.018 → stationary |
| 5 | ACF/PACF inspection | Inconclusive → use auto.arima |
| 6 | auto.arima (23 candidates) | Best: SARIMA(2,0,3)(1,1,0)[4], AICc = −168.15 |
| 7 | Diagnostic checking | Clean residuals, Ljung-Box p > 0.05 ✓ |
| 8 | 12-step forecast | Q1 1987 – Q4 1989, seasonal pattern preserved |

**Final Model:** SARIMA(2,0,3)(1,1,0)[4] on log-transformed series

$$
(1 - 0.218B - 0.764B^2)(1 - 0.181B^4)(1-B^4)\log(X_t) = (1 - 0.321B - 0.807B^2 + 0.337B^3)W_t
$$

**Methods:** Log transformation, seasonal differencing, ADF test, ACF/PACF, `auto.arima`, `sarima` diagnostics, forecast with prediction intervals

---

### 2. ARIMA Analysis of Monthly US Chicken Prices
**`chicken_prices_arima.R`**

Analysis of the `chicken` dataset from the `astsa` package covering monthly chicken prices (cents per pound) from January 2001 to December 2016 (192 monthly observations).

**Workflow:**

| Step | Action | Result |
|------|--------|--------|
| 1 | Plot raw series | Clear upward trend; stable variance → no log transform |
| 2 | ADF test on raw | p = 0.411 → non-stationary |
| 3 | First differencing | Series fluctuates around zero |
| 4 | ADF test on differenced | p < 0.01 → stationary; no seasonal diff needed |
| 5 | ACF/PACF of differenced | AR signature; seasonal signal at lag 12 |
| 6 | auto.arima | Best: ARIMA(3,1,0)(0,0,1)[12], AICc = 1.96 |
| 7 | Diagnostic checking | Best Ljung-Box performance of all candidates |

**Final Model:** ARIMA(3,1,0)(0,0,1)[12]

$$
(1 - 0.898B + 0.142B^2 + 0.126B^3)(1-B)X_t = (1 + 0.290B^{12})W_t, \quad W_t \sim WN(0, 0.389)
$$

**Methods:** ADF testing, first differencing, ACF/PACF, `auto.arima`, model comparison by AICc, `sarima` diagnostics

---

## Model Comparison Highlights

| Dataset | Final Model | AICc | Key Technique |
|---------|-------------|------|---------------|
| UK Gas (quarterly) | SARIMA(2,0,3)(1,1,0)[4] | −168.15 | Log + seasonal diff + seasonal AR |
| Chicken (monthly) | ARIMA(3,1,0)(0,0,1)[12] | 1.9612 | First diff + seasonal MA |

---

## Tech Stack

- **Language:** R
- **Key Packages:** `forecast`, `astsa`, `tseries`
- **Core Functions:** `auto.arima()`, `sarima()`, `adf.test()`, `acf()`, `pacf()`, `forecast()`

---

## Repository Structure

```
time-series-analysis/
├── uk_gas_sarima.R           # SARIMA analysis of UK quarterly gas consumption
├── chicken_prices_arima.R    # ARIMA analysis of monthly US chicken prices
└── README.md
```

---

## Author

**Chris Harry** — Master of Data Science

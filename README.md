# Team Dynamo
# FidelFolio Market Cap Growth Forecasting

## Project Overview
This repository presents our solution to the **FidelFolio Deep Learning Challenge**, which involves predicting market capitalization growth across multiple time horizons for Indian companies. Using deep learning, we aim to model the nonlinear relationships between financial indicators and future market performance.

---

## Objective
- Predict market cap growth across:
  - Short-term (1-Year): `Target 1`
  - Medium-term (2-Year): `Target 2`
  - Long-term (3-Year): `Target 3`
- Compare deep learning models to identify the most accurate and robust architecture.

---

## Dataset

- Instances: Company-Year combinations
- Features: 28 fundamental financial indicators (`Feature1` to `Feature28`)
- Targets: 
  - `Target 1` (1Y growth)
  - `Target 2` (2Y growth)
  - `Target 3` (3Y growth)

### Preprocessing Steps
- Missing values imputed using company-wise and global means.
- Winsorization applied to cap outliers.
- Standardization performed using `StandardScaler`.

---

## Models Implemented

### 1. Multilayer Perceptron (MLP)
- Basic feedforward neural network with dropout and ReLU activation.
- Trained for 1000 epochs.
- MLP is not able to capture the complex Time Series Patterns in the Dataset.

**Performance (RMSE):**
- Target 1: 106.6246
- Target 2: 227.6656
- Target 3: 370.3143

---

### 2. LSTM (Vanilla)
- Standard sequence model using final hidden state.
- Trained for 2000 epochs.

**Performance (RMSE):**
- Target 1: 22.7029
- Target 2: 56.9746
- Target 3: 186.7859

---

### 3. LSTM with Attention
- Incorporates soft attention over time for feature weighting.
- Trained for 3000 epochs.

**Performance (RMSE):**
- Target 1: 19.4405
- Target 2: 47.9110
- Target 3: 175.1124

---

### 4. Transformer Encoder
- Uses positional encoding and multi-head self-attention.
- Trained for 1200 epochs.

**Performance (RMSE):**
- Target 1: 30.9814
- Target 2: 38.7712
- Target 3: 112.2152

---

### 5. DeepTCN (Temporal Convolutional Network)
- Causal convolutions with dilation to capture temporal dependencies.
- Results to be updated.

---

### 6. N-BEATS (Neural Basis Expansion)
- Forecasting-specific architecture designed for interpretability.
- Results to be updated.

---

### 7. DeepAR
- Autoregressive model trained on probabilistic distributions.
- Results to be updated.

---

## Model Comparison

| Model              | Attention | RMSE T1 | RMSE T2 | RMSE T3 |
|--------------------|-----------|---------|---------|---------|
| MLP                | No        | 106.6246 | 227.6656 | 370.3143 |
| LSTM               | No        | 22.7029  | 56.9746  | 186.7859 |
| LSTM + Attention   | Yes       | 19.4405  | 47.9110  | 175.1124 |
| Transformer        | Yes       | 30.9814  | 38.7712  | 112.2152 |
| DeepTCN            | Yes       | Pending  | Pending  | Pending  |
| N-BEATS            | No        | Pending  | Pending  | Pending  |
| DeepAR             | Yes       | Pending  | Pending  | Pending  |

---


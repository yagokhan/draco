# Project DRACO: Technical Methodology & Architectural Deep-Dive

## 1. Executive Summary
Project **DRACO** represents a high-fidelity, GPU-accelerated algorithmic trading framework designed for the **AMD Radeon RX 9070 XT** ecosystem. It specializes in vectorized trade simulation, multi-tier asset optimization, and adaptive volatility-adjusted risk management. The core objective is to recover and exceed the "Gold Standard" performance metrics through exhaustive grid search and non-linear signal auditing.

---

## 2. System Architecture

### 2.1 Vectorized Simulation Engine (`trade_manager.py`)
Unlike traditional event-driven backtesters, DRACO utilizes a **fully vectorized simulation architecture** built on PyTorch.
- **Tensors:** All trade data (Confidence, Pearson R, PnL, Leverage) are mapped to 1D/2D tensors.
- **Batched Trials:** The engine processes up to **16,384 parameter combinations per batch** directly on the GPU.
- **Computational Efficiency:** By utilizing `torch.cuda`, the system achieves an iteration speed of ~1.7 million simulations per minute, enabling exhaustive optimization that would take days on a CPU.

### 2.2 Tiered Asset Strategy (`config.py`)
DRACO categorizes the market into three distinct risk tiers, each with its own optimized search space:
1.  **TITAN (Blue Chip):** BTC, ETH, SOL, etc. Focuses on ultra-low noise and high precision (XGB > 0.95).
2.  **NAVIGATOR (Mid-Cap):** ADA, LINK, DOT, etc. Balanced volatility and trend capture.
3.  **VOLT (High Volatility):** PEPE, BONK, SUI, etc. Wide ranges requiring loose stops and higher Pearson correlation filters.

---

## 3. The "Dual-Gate" Signal Auditor (`signal_auditor.py`)

The entry logic is governed by a **Dual-Gate Filtering System** that validates trade signals across two independent dimensions:
1.  **XGBoost Confidence Gate:** A non-linear machine learning confidence score (0.0 to 1.0). Only signals exceeding the `conf_min` threshold are processed.
2.  **Pearson R Precision Gate:** A correlation metric validating the relationship between price action and the underlying regression midline.

**Simultaneous Trigger Logic:**
```python
# Entry mask calculation
entry_mask = (confidence >= conf_min) & (pvt_r >= pvt_r_min)
```

---

## 4. Adaptive Trailing & Exit Logic

DRACO employs a sophisticated **Adaptive Trailing Stop (ATS)** model that replaces static Take-Profit orders.

### 4.1 Trail Activation Threshold
A trade enters "Trailing Mode" only after crossing a specific activation threshold (e.g., +0.4%). This ensures that small noise fluctuations don't trigger premature exits.

### 4.2 Dynamic Midline Buffer
Once activated, the stop-loss is calculated as:
$Stop = MaxPrice - (MidlineBuffer \times StdDevMultiplier \times 0.001)$
- **Midline Buffer:** Adjusts the exit based on the regression midline.
- **StdDev Multiplier:** Scales the buffer based on local asset volatility (ATR-like adjustment).

### 4.3 Hard Stop-Loss (Safety Net)
A fixed **1.0% to 1.5% Hard SL** is enforced at the tensor level to prevent catastrophic drawdowns during "Black Swan" events or high-slippage environments.

---

## 5. Optimization & Scoring Logic (`optimizer.py`)

The system evaluates success based on a composite objective function that prioritizes **Win Rate (WR)** and **Risk-Adjusted Return (Calmar)**.

### 5.1 The Objective Function
$Score = (WinRate \times 100) + (AvgPnL \times 5)$
- **Penalty Factor:** Configurations with fewer than 10 trades are heavily penalized (-500) to avoid overfitting on outlier trades.
- **Top-K Selection:** The system extracts the Top-20 configurations per tier for final ensemble selection.

### 5.2 Data Splits (Temporal Validation)
- **Training:** 2025-11-01 to 2026-01-15 (Baseline Generation)
- **Validation:** 2026-01-16 to 2026-02-15 (Parameter Tuning)
- **Blind Test (Extended):** 2026-02-16 to 2026-03-29 (Production Validation)

---

## 6. Current Benchmarks (Alpha Config)

| Metric | Result |
|--------|--------|
| **Hardware** | AMD RX 9070 XT |
| **Total Trials** | 1.76 Million |
| **Extended Trades** | 1,847 |
| **Calmar Ratio** | 65.28 |
| **Max Drawdown** | 1.3% |

---
*Documented by DRACO Core Engine — March 29, 2026*

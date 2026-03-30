"""
Project DRACO — 50-Asset Institutional Engine (KRYPTONITE V5).
"""
from __future__ import annotations
import torch
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DRACO_CSV = PROJECT_ROOT / "draco_universe_50.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
DTYPE = torch.float32

# TITAN (10), NAVIGATOR (20), VOLT (20)
TITAN_ASSETS = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "LINK", "DOT", "LTC", "BCH"]
NAVIGATOR_ASSETS = ["MATIC", "AVAX", "NEAR", "ATOM", "FIL", "ICP", "STX", "RNDR", "FET", "GRT", 
                    "INJ", "THETA", "AR", "OP", "ARB", "TIA", "SEI", "SUI", "APT", "VET"]
VOLT_ASSETS = ["PEPE", "SHIB", "DOGE", "BONK", "WIF", "FLOKI", "MEME", "ORDI", "1000SATS", "AAVE",
               "MKR", "LDO", "UNI", "DYDX", "CRV", "IMX", "PYTH", "JUP", "ENA", "W"]

ASSET_TO_TIER = {a: 0 for a in TITAN_ASSETS}
ASSET_TO_TIER.update({a: 1 for a in NAVIGATOR_ASSETS})
ASSET_TO_TIER.update({a: 2 for a in VOLT_ASSETS})

TF_ENCODE = {"5m": 0, "30m": 1, "1h": 2, "4h": 3}

# --- DRACO 2-Year Timeline ---
TRAIN_START = "2024-01-01"
TRAIN_END   = "2025-06-30"
VAL_START   = "2025-07-01"
VAL_END     = "2025-10-31"
BLIND_START = "2025-11-01"
BLIND_END   = "2026-03-29"

# Real-World Institutional Friction
ROUND_TRIP_COST = 0.0045 # 0.12% fees + 0.33% slippage/spread
INITIAL_CAPITAL = 1000.0
LATENCY_PENALTY = 0.0015 # 0.15% worse entry/exit prices

# Institutional Calibration
POS_FRAC_BASE    = 0.12
CONF_WEIGHT_POW  = 3.0    # Strict weight
PTP_TARGET_PCT   = 2.50   # 2.5% Target
PTP_EXIT_FRAC    = 0.5    # 50% Exit

@dataclass
class TierSearchSpace:
    name: str
    conf_min:     tuple
    pvt_r_min:    tuple
    midline_buf:  tuple
    stddev_mult:  tuple
    hard_sl:      tuple
    activation:   tuple = (0.005, 0.012, 0.002)

# TITAN: High-Confidence Institutional Search
TITAN_SPACE = TierSearchSpace(
    name="TITAN",
    conf_min=(0.600, 0.950, 0.020),
    pvt_r_min=(0.550, 0.900, 0.020),
    midline_buf=(0.05, 0.35, 0.05),
    stddev_mult=(0.5, 1.5, 0.2),
    hard_sl=(0.004, 0.008, 0.001)
)

# NAVIGATOR
NAVIGATOR_SPACE = TierSearchSpace(
    name="NAVIGATOR",
    conf_min=(0.550, 0.900, 0.020),
    pvt_r_min=(0.500, 0.850, 0.020),
    midline_buf=(0.10, 0.40, 0.05),
    stddev_mult=(1.0, 2.5, 0.2),
    hard_sl=(0.008, 0.018, 0.002)
)

# VOLT
VOLT_SPACE = TierSearchSpace(
    name="VOLT",
    conf_min=(0.500, 0.850, 0.020),
    pvt_r_min=(0.450, 0.800, 0.020),
    midline_buf=(0.20, 0.70, 0.10),
    stddev_mult=(2.0, 5.0, 0.5),
    hard_sl=(0.015, 0.045, 0.005)
)

GPU_BATCH_SIZE = 4096

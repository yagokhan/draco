"""
Project DRACO — 50-Asset Robust Sniper Engine (KRYPTONITE EDITION).
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

ROUND_TRIP_COST = 0.0012
INITIAL_CAPITAL = 15_000.0

# Aggressive but Precision-Focused
POS_FRAC_BASE    = 0.12
CONF_WEIGHT_POW  = 3.0
PTP_TARGET_PCT   = 1.50
PTP_EXIT_FRAC    = 0.5

@dataclass
class TierSearchSpace:
    name: str
    conf_min:     tuple
    pvt_r_min:    tuple
    midline_buf:  tuple
    stddev_mult:  tuple
    hard_sl:      tuple
    activation:   tuple = (0.004, 0.010, 0.001)

# TITAN: Robust Search
TITAN_SPACE = TierSearchSpace(
    name="TITAN",
    conf_min=(0.650, 0.950, 0.050),
    pvt_r_min=(0.600, 0.900, 0.050),
    midline_buf=(0.05, 0.35, 0.05),
    stddev_mult=(0.5, 1.5, 0.2),
    hard_sl=(0.003, 0.008, 0.001)
)

# NAVIGATOR
NAVIGATOR_SPACE = TierSearchSpace(
    name="NAVIGATOR",
    conf_min=(0.600, 0.900, 0.050),
    pvt_r_min=(0.550, 0.850, 0.050),
    midline_buf=(0.10, 0.40, 0.05),
    stddev_mult=(0.8, 2.2, 0.2),
    hard_sl=(0.006, 0.018, 0.002)
)

# VOLT
VOLT_SPACE = TierSearchSpace(
    name="VOLT",
    conf_min=(0.550, 0.850, 0.050),
    pvt_r_min=(0.500, 0.800, 0.050),
    midline_buf=(0.20, 0.60, 0.10),
    stddev_mult=(1.5, 4.0, 0.5),
    hard_sl=(0.010, 0.040, 0.005)
)

GPU_BATCH_SIZE = 4096

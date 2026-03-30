"""
draco/apply_signals.py — Update DRACO Universe with XGBoost Confidence.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from config import DRACO_CSV
import logging

PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"

logger = logging.getLogger("draco_apply")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")

FEATURE_COLS = ["confidence", "pvt_r", "rsi", "ema_dist", "mom"]
TF_ENCODE = {"5m": 0, "30m": 1, "1h": 2, "4h": 3}

def apply():
    logger.info(f"Loading DRACO model and universe...")
    model = xgb.Booster()
    model.load_model(str(MODELS_DIR / "draco_meta_xgb.json"))
    
    df = pd.read_csv(DRACO_CSV)
    df["best_tf_encoded"] = df["best_tf"].map(TF_ENCODE)
    
    X = df[FEATURE_COLS].copy()
    dmatrix = xgb.DMatrix(X)
    
    logger.info(f"Predicting confidence for {len(df)} signals...")
    df["xgb_confidence"] = model.predict(dmatrix)
    
    # Replace raw confidence with XGB confidence for the optimizer
    # Keep original as 'raw_r'
    df["raw_r"] = df["confidence"]
    df["confidence"] = df["xgb_confidence"]
    
    df.to_csv(DRACO_CSV, index=False)
    logger.info(f"DRACO UNIVERSE UPDATED: Confidence scores applied.")

if __name__ == "__main__":
    apply()

"""
draco/train_draco_xgb.py — Purged 5-Fold XGBoost Trainer for DRACO V7.
"""
from __future__ import annotations
import torch
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import KFold # Standard KFold for chronological purging
from sklearn.metrics import roc_auc_score
from config import DRACO_CSV, TRAIN_START, TRAIN_END
import logging

PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

DRACO_XGB_PARAMS = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "device": "cpu",
    "max_depth": 6,
    "learning_rate": 0.03,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1
}

FEATURE_COLS = ["confidence", "pvt_r", "rsi", "ema_dist", "mom"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")
logger = logging.getLogger("draco_xgb_purged")

def train_draco_model(data_path: Path, n_folds: int = 5, purge_bars: int = 12):
    logger.info(f"Loading data for PURGED DRACO training: {data_path}")
    df = pd.read_csv(data_path)
    df["entry_date"] = pd.to_datetime(df["entry_ts"]).dt.strftime("%Y-%m-%d")
    train_df = df[(df.entry_date >= TRAIN_START) & (df.entry_date <= TRAIN_END)].copy()
    
    # Chronological sort for proper purging
    train_df = train_df.sort_values("entry_ts")
    
    kf = KFold(n_splits=n_folds, shuffle=False) # Chronological split
    fold_aucs = []
    
    X = train_df[FEATURE_COLS].values
    y = (train_df["pnl_usd"] > 0).astype(int).values
    
    logger.info(f"Starting {n_folds}-Fold PURGED Cross-Validation (Gap: {purge_bars}h)...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        # Apply Purging: Remove indices near the boundaries
        # We remove purge_bars from the end of the training set if it precedes validation
        # or from the beginning of training if it follows validation.
        
        val_start_idx = val_idx[0]
        val_end_idx = val_idx[-1]
        
        # Purge Training set
        purged_train_idx = [i for i in train_idx if i < val_start_idx - purge_bars or i > val_end_idx + purge_bars]
        
        X_train, X_val = X[purged_train_idx], X[val_idx]
        y_train, y_val = y[purged_train_idx], y[val_idx]
        
        model = xgb.XGBClassifier(**DRACO_XGB_PARAMS)
        model.fit(X_train, y_train)
        
        preds = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)
        fold_aucs.append(auc)
        logger.info(f"  Fold {fold+1}/{n_folds}: Purged AUC = {auc:.4f}")
    
    logger.info(f"Mean Purged CV AUC: {np.mean(fold_aucs):.4f}")
    
    # Final Model
    final_model = xgb.XGBClassifier(**DRACO_XGB_PARAMS)
    final_model.fit(X, y)
    final_model.get_booster().save_model(str(MODELS_DIR / "draco_meta_xgb.json"))
    logger.info(f"Final PURGED model saved.")

if __name__ == "__main__":
    train_draco_model(DRACO_CSV)

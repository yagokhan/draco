"""
draco/train_draco_xgb.py — 5-Fold GPU-Accelerated XGBoost Trainer for DRACO.
"""
from __future__ import annotations
import torch
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from config import DRACO_CSV
import logging

PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# CPU Optimized Params (switching back to ensure stability)
DRACO_XGB_PARAMS = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "device": "cpu",
    "max_depth": 6,
    "learning_rate": 0.03,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.2,
    "random_state": 42,
    "n_jobs": -1
}

FEATURE_COLS = ["confidence", "pvt_r", "rsi", "ema_dist", "mom"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")
logger = logging.getLogger("draco_xgb")

def train_draco_model(data_path: Path, n_folds: int = 5):
    logger.info(f"Loading data for 5-FOLD DRACO training: {data_path}")
    df = pd.read_csv(data_path)
    
    # Target: Profitable trade (PnL > 0)
    y = (df["pnl_usd"] > 0).astype(int).values
    X = df[FEATURE_COLS].copy()
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_aucs = []
    
    logger.info(f"Starting {n_folds}-Fold Cross-Validation...")
    
    X_np = X.values
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_np, y)):
        X_train, X_val = X_np[train_idx], X_np[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = xgb.XGBClassifier(**DRACO_XGB_PARAMS)
        model.fit(X_train, y_train)
        
        preds = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)
        fold_aucs.append(auc)
        logger.info(f"  Fold {fold+1}/{n_folds}: AUC = {auc:.4f}")
    
    logger.info(f"Mean CV AUC: {np.mean(fold_aucs):.4f} (+/- {np.std(fold_aucs):.4f})")
    
    # Final Model on all data
    final_model = xgb.XGBClassifier(**DRACO_XGB_PARAMS)
    final_model.fit(X_np, y)
    
    final_model.get_booster().save_model(str(MODELS_DIR / "draco_meta_xgb.json"))
    logger.info(f"Final model saved to draco/models/draco_meta_xgb.json")

if __name__ == "__main__":
    train_draco_model(DRACO_CSV)

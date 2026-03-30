"""
draco/generate_draco_universe.py — Audit-Ready Synthesis for 50-Asset DRACO.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
import logging

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DRACO_CSV = PROJECT_ROOT / "draco_universe_50.csv"

logger = logging.getLogger("draco_gen")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")

# 50 Assets
ASSETS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "LINKUSDT", "DOTUSDT", "LTCUSDT", "BCHUSDT",
    "MATICUSDT", "AVAXUSDT", "NEARUSDT", "ATOMUSDT", "FILUSDT", "ICPUSDT", "STXUSDT", "RNDRUSDT", "FETUSDT", "GRTUSDT",
    "INJUSDT", "THETAUSDT", "ARUSDT", "OPUSDT", "ARBUSDT", "TIAUSDT", "SEIUSDT", "SUIUSDT", "APTUSDT", "VETUSDT",
    "PEPEUSDT", "SHIBUSDT", "DOGEUSDT", "BONKUSDT", "WIFUSDT", "FLOKIUSDT", "MEMEUSDT", "ORDIUSDT", "1000SATSUSDT", "AAVEUSDT",
    "MKRUSDT", "LDOUSDT", "UNIUSDT", "DYDXUSDT", "CRVUSDT", "IMXUSDT", "PYTHUSDT", "JUPUSDT", "ENAUSDT", "WUSDT"
]

def _compute_rolling_r(y: pd.Series, period: int):
    x = pd.Series(np.arange(period), index=y.index[:period])
    return y.rolling(window=period).corr(pd.Series(np.arange(len(y)), index=y.index))

def generate():
    logger.info(f"Generating AUDIT-READY DRACO universe...")
    all_signals = []
    
    for symbol in ASSETS:
        p = DATA_DIR / f"{symbol}_1h.parquet"
        if not p.exists(): continue
        
        df = pd.read_parquet(p)
        if len(df) < 100: continue
        
        c = df["close"]
        v = df["volume"].replace(0, 1e-9)
        
        # 1. Indicators
        df["rsi"] = 50.0
        try:
            delta = c.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            df["rsi"] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))
        except: pass
        
        df["ema_20"] = c.ewm(span=20).mean()
        df["mom"] = c.pct_change(12)
        
        # 2. Key Regression Filter
        df["confidence_raw"] = np.log(c).rolling(48).corr(pd.Series(np.arange(len(c)), index=c.index)).abs()
        df["pvt_r_val"] = np.log(c).rolling(48).corr(np.log(v)).abs()
        
        # 3. Prices and Target
        df["entry_price"] = c
        df["exit_price"] = c.shift(-12)
        df["pnl_pct"] = (df["exit_price"] / df["entry_price"] - 1.0) * 100.0
        
        df = df.replace([np.inf, -np.inf], 0).fillna(0)
        df_sample = df.iloc[::4].copy()
        
        signals = pd.DataFrame({
            "asset": symbol.replace("USDT", ""),
            "entry_ts": df_sample.index,
            "confidence": np.clip(df_sample["confidence_raw"], 0, 1),
            "pvt_r": np.clip(df_sample["pvt_r_val"], 0, 1),
            "rsi": np.clip(df_sample["rsi"], 0, 100),
            "ema_dist": np.clip((df_sample["close"] / df_sample["ema_20"] - 1.0), -0.5, 0.5),
            "mom": np.clip(df_sample["mom"], -0.5, 0.5),
            "buy_price": df_sample["entry_price"],
            "sell_price": df_sample["exit_price"],
            "best_tf": "1h",
            "best_period": 48,
            "pnl_pct": df_sample["pnl_pct"],
            "pnl_usd": df_sample["pnl_pct"] * 10.0,
            "leverage": 3.0,
            "bars_held": 12
        })
        
        all_signals.append(signals)
        logger.info(f"  {symbol}: {len(signals)} signals")

    if all_signals:
        master_df = pd.concat(all_signals, ignore_index=True)
        master_df.to_csv(DRACO_CSV, index=False)
        logger.info(f"DRACO AUDIT UNIVERSE READY: {len(master_df)} signals")

if __name__ == "__main__":
    generate()

#!/usr/bin/env python3
"""
draco/data_fetcher.py — 50-Asset Hyper-Aggressive Data Engine.
"""
from __future__ import annotations
import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"

# TITAN (10), NAVIGATOR (20), VOLT (20) = 50 Assets
ASSETS = [
    # TITAN
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "LINKUSDT", "DOTUSDT", "LTCUSDT", "BCHUSDT",
    # NAVIGATOR
    "MATICUSDT", "AVAXUSDT", "NEARUSDT", "ATOMUSDT", "FILUSDT", "ICPUSDT", "STXUSDT", "RNDRUSDT", "FETUSDT", "GRTUSDT",
    "INJUSDT", "THETAUSDT", "ARUSDT", "OPUSDT", "ARBUSDT", "TIAUSDT", "SEIUSDT", "SUIUSDT", "APTUSDT", "VETUSDT",
    # VOLT
    "PEPEUSDT", "SHIBUSDT", "DOGEUSDT", "BONKUSDT", "WIFUSDT", "FLOKIUSDT", "MEMEUSDT", "ORDIUSDT", "1000SATSUSDT", "AAVEUSDT",
    "MKRUSDT", "LDOUSDT", "UNIUSDT", "DYDXUSDT", "CRVUSDT", "IMXUSDT", "PYTHUSDT", "JUPUSDT", "ENAUSDT", "WUSDT"
]

TIMEFRAMES = ["5m", "30m", "1h", "4h"]
TF_MS = {"5m": 300000, "30m": 1800000, "1h": 3600000, "4h": 14400000}

# DRACO 2-Year Range
GLOBAL_START = datetime(2024, 1, 1, tzinfo=timezone.utc)
GLOBAL_END   = datetime(2026, 3, 29, 23, 59, 59, tzinfo=timezone.utc)

KLINES_URL  = "https://fapi.binance.com/fapi/v1/klines"
SPOT_URL    = "https://api.binance.com/api/v3/klines"
MAX_CANDLES = 1000
RATE_LIMIT_S = 0.1

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")
logger = logging.getLogger("draco_fetcher")

def _ts_ms(dt): return int(dt.timestamp() * 1000)
def _parquet_path(symbol, tf): return DATA_DIR / f"{symbol}_{tf}.parquet"

def _fetch_klines(symbol, interval, start_ms, end_ms):
    all_klines = []
    cursor = start_ms
    url = KLINES_URL
    while cursor < end_ms:
        params = {"symbol": symbol, "interval": interval, "startTime": cursor, "endTime": end_ms, "limit": MAX_CANDLES}
        for attempt in range(5):
            try:
                logger.info(f"    Fetching batch for {symbol} {interval} (cursor: {cursor})")
                resp = requests.get(url, params=params, timeout=10)
                if resp.status_code == 429: 
                    logger.warning("    Rate limited (429). Sleeping 30s...")
                    time.sleep(30); continue
                if resp.status_code == 400 and url == KLINES_URL: 
                    logger.warning(f"    Futures 400 for {symbol}, trying Spot...")
                    url = SPOT_URL; continue
                resp.raise_for_status()
                batch = resp.json()
                break
            except Exception as e:
                logger.error(f"    Attempt {attempt+1} failed: {e}")
                time.sleep(2**attempt); continue
        else: break
        if not batch: break
        all_klines.extend(batch)
        cursor = int(batch[-1][0]) + TF_MS[interval]
        time.sleep(RATE_LIMIT_S)
    return all_klines

def _klines_to_df(raw):
    if not raw: return pd.DataFrame()
    df = pd.DataFrame(raw, columns=["open_time", "open", "high", "low", "close", "volume", "ct", "qv", "tr", "tb", "tq", "i"])
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    for col in ("open", "high", "low", "close", "volume"): df[col] = df[col].astype(float)
    return df[["open", "high", "low", "close", "volume"]].sort_index()

def fetch_symbol_tf(symbol, tf):
    path = _parquet_path(symbol, tf)
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    start_ms = _ts_ms(GLOBAL_START)
    if not existing.empty:
        start_ms = int(existing.index[-1].timestamp() * 1000) + TF_MS[tf]
    
    if start_ms >= _ts_ms(GLOBAL_END):
        logger.info(f"  {symbol} {tf} UP TO DATE")
        return
    
    raw = _fetch_klines(symbol, tf, start_ms, _ts_ms(GLOBAL_END))
    new_df = _klines_to_df(raw)
    if not new_df.empty:
        combined = pd.concat([existing, new_df]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.to_parquet(path, engine="pyarrow")
        logger.info(f"  {symbol} {tf} SAVED {len(combined)} bars")

if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"DRACO FETCH START: {len(ASSETS)} assets")
    for symbol in ASSETS:
        for tf in TIMEFRAMES:
            fetch_symbol_tf(symbol, tf)

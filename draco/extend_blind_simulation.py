"""
Project DRACO — Extended Blind Simulation.
Extends the blind test period to Mar 29, 2026, by merging all trade sources.
"""
from __future__ import annotations
import torch
import pandas as pd
import json
from pathlib import Path
from config import (
    DEVICE, DTYPE, GOLD_CSV, ASSET_TO_GROUP, TF_ENCODE,
    TRAIN_START, TRAIN_END, VAL_START, VAL_END, BLIND_START,
)
from signal_auditor import apply_entry_gates, apply_exit_strategy

EXTENDED_END = "2026-03-29"

def load_extended_universe() -> pd.DataFrame:
    # 1. Load Gold Standard
    df_gold = pd.read_csv(GOLD_CSV)
    
    # 2. Load Restored Extended (for gap filling)
    RAW_CSV = Path(GOLD_CSV).parent / "blind_test_v1_restored_trades.csv"
    df_raw = pd.read_csv(RAW_CSV)
    
    # 3. Load Live Trades (for recent dates Mar 26 - 29)
    LIVE_CSV = Path(GOLD_CSV).parent / "live_extended_v1_trades.csv"
    df_live = pd.read_csv(LIVE_CSV)
    
    # Normalize and merge
    df = pd.concat([df_gold, df_raw, df_live], ignore_index=True)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], format='ISO8601', utc=True)
    df = df.sort_values("entry_ts").drop_duplicates(subset=["asset", "entry_ts"], keep="first").reset_index(drop=True)
    
    return df

def run_extended_blind():
    df = load_extended_universe()
    
    # Filter for Blind period: Feb 16 to Mar 29
    mask = (df.entry_ts >= BLIND_START) & (df.entry_ts <= EXTENDED_END)
    df_blind = df[mask].copy()
    
    # Load best params
    with open("results/draco_gold_config.json", "r") as f:
        best = json.load(f)
    
    params = torch.tensor([[
        best["conf_min"], best["pvt_r_min"], 
        best["midline_buffer"], best["stddev_mult"], 
        best["trail_activation"], 0.015
    ]], dtype=DTYPE, device=DEVICE)
    
    # Convert to tensors
    data = {
        "n": len(df_blind),
        "confidence": torch.tensor(df_blind["confidence"].values, dtype=DTYPE, device=DEVICE),
        "pvt_r":      torch.tensor(df_blind["pvt_r"].values, dtype=DTYPE, device=DEVICE),
        "pnl_pct":    torch.tensor(df_blind["pnl_pct"].values, dtype=DTYPE, device=DEVICE),
        "leverage":   torch.tensor(df_blind["leverage"].values, dtype=DTYPE, device=DEVICE),
        "bars_held":  torch.tensor(df_blind["bars_held"].values, dtype=DTYPE, device=DEVICE),
        "tf_id":      torch.tensor(df_blind["best_tf"].map(TF_ENCODE).values, dtype=torch.int32, device=DEVICE),
    }
    
    # Simulate
    entry_mask = apply_entry_gates(data, params)
    adj_pnl_pct = apply_exit_strategy(data, params, entry_mask)
    
    # Final filter
    executed = entry_mask[0].cpu().numpy()
    final_pnl = adj_pnl_pct[0].cpu().numpy()
    
    df_blind["executed"] = executed
    df_blind["final_pnl_pct"] = final_pnl
    
    # Results
    df_exec = df_blind[df_blind["executed"]].copy()
    
    print(f"\nEXTENDED BLIND TEST: {BLIND_START} to {EXTENDED_END}")
    print(f"Total Trades Executed: {len(df_exec)}")
    print(f"Win Rate: { (df_exec['final_pnl_pct'] > 0).mean():.1%}")
    print(f"Avg PnL: {df_exec['final_pnl_pct'].mean():.2f}%")
    
    print("\nLATEST TRADES (Mar 25 - Mar 29):")
    latest = df_exec[df_exec.entry_ts >= "2026-03-25"].sort_values("entry_ts")
    print(latest[["entry_ts", "asset", "direction", "confidence", "pvt_r", "final_pnl_pct"]].to_string(index=False))

    # Save to CSV for the user
    df_exec.to_csv("blind_test_extended_today.csv", index=False)
    print(f"\nFull extended trade list saved to blind_test_extended_today.csv")

if __name__ == "__main__":
    run_extended_blind()

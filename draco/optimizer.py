"""
Project DRACO — 5-Fold Cross-Validated GPU Optimizer.
"""
from __future__ import annotations
import torch
import time
import json
import numpy as np
from pathlib import Path
from config import (
    DEVICE, DTYPE, GPU_BATCH_SIZE, RESULTS_DIR,
    TITAN_SPACE, NAVIGATOR_SPACE, VOLT_SPACE,
    TRAIN_START, TRAIN_END
)
from trade_manager import simulate_batch, extract_top_k, BatchResult
from signal_auditor import load_trade_universe

def build_grid(space) -> torch.Tensor:
    c = torch.arange(space.conf_min[0], space.conf_min[1]+0.0001, space.conf_min[2], dtype=DTYPE, device=DEVICE)
    p = torch.arange(space.pvt_r_min[0], space.pvt_r_min[1]+0.0001, space.pvt_r_min[2], dtype=DTYPE, device=DEVICE)
    b = torch.arange(space.midline_buf[0], space.midline_buf[1]+0.0001, space.midline_buf[2], dtype=DTYPE, device=DEVICE)
    s = torch.arange(space.stddev_mult[0], space.stddev_mult[1]+0.0001, space.stddev_mult[2], dtype=DTYPE, device=DEVICE)
    a = torch.arange(space.activation[0], space.activation[1]+0.0001, space.activation[2], dtype=DTYPE, device=DEVICE)
    h = torch.arange(space.hard_sl[0], space.hard_sl[1]+0.0001, space.hard_sl[2], dtype=DTYPE, device=DEVICE)
    
    grid = torch.meshgrid(c, p, b, s, a, h, indexing="ij")
    params = torch.stack([g.flatten() for g in grid], dim=1)
    return params

def run_tier_optimization():
    data_splits = load_trade_universe()
    best_configs = {}

    for space in [TITAN_SPACE, NAVIGATOR_SPACE, VOLT_SPACE]:
        print(f"\n[V6-CV] Optimizing Tier: {space.name}")
        tier_id = {"TITAN": 0, "NAVIGATOR": 1, "VOLT": 2}[space.name]
        
        # 1. Prepare 5 Folds from the Training Data
        train_full = data_splits["train"]
        mask = (train_full["tier_id"] == tier_id)
        train_tier = {k: v[mask] if isinstance(v, torch.Tensor) and v.shape[0] > 0 else v for k, v in train_full.items()}
        n_total = mask.sum().item()
        
        if n_total < 500:
            print(f"  Insufficient trades for {space.name}")
            continue

        params = build_grid(space)
        N_grid = params.shape[0]
        print(f"  Grid Size: {N_grid:,} trials | Total Trades: {n_total}")
        
        # Create Fold Indices
        fold_size = n_total // 5
        fold_results = [] # Will store BatchResults for each fold

        for f_idx in range(5):
            f_start, f_end = f_idx * fold_size, (f_idx + 1) * fold_size
            fold_data = {k: v[f_start:f_end] if isinstance(v, torch.Tensor) and v.shape[0] > 0 else v for k, v in train_tier.items()}
            fold_data["n"] = f_end - f_start
            
            fold_batch_pnl = []
            fold_batch_score = []
            
            for i in range(0, N_grid, GPU_BATCH_SIZE):
                batch = params[i:i+GPU_BATCH_SIZE]
                res = simulate_batch(fold_data, batch)
                fold_batch_pnl.append(res.net_pnl)
                fold_batch_score.append(res.score)
            
            fold_results.append({
                "pnl": torch.cat(fold_batch_pnl),
                "score": torch.cat(fold_batch_score)
            })
            print(f"    Fold {f_idx+1} processed.")

        # 2. Find the configuration with the highest MEAN SCORE across all 5 folds
        all_scores = torch.stack([f["score"] for f in fold_results], dim=0) # [5, N_grid]
        mean_scores = all_scores.mean(dim=0)
        
        # We also calculate variance to penalize unstable parameters
        std_scores = all_scores.std(dim=0)
        cv_score = mean_scores - (std_scores * 0.5) # Penalty for inconsistency
        
        best_idx = torch.argmax(cv_score).item()
        p = params[best_idx]
        
        best_configs[space.name] = {
            "conf_min": p[0].item(),
            "pvt_r_min": p[1].item(),
            "midline_buffer": p[2].item(),
            "stddev_mult": p[3].item(),
            "trail_activation": p[4].item(),
            "hard_sl": p[5].item(),
            "cv_mean_score": mean_scores[best_idx].item()
        }
        print(f"  Best {space.name} (CV Optimized): Conf={p[0].item():.3f} | SL={p[5].item():.1%}")

    # 3. Final Validation on Blind Test
    print("\n" + "="*80)
    print("  DRACO V6 CROSS-VALIDATED PERFORMANCE (Blind Test)")
    print("="*80)
    
    blind_full = data_splits["blind"]
    total_trades = 0; total_wins = 0; total_pnl = 0.0
    
    for name, cfg in best_configs.items():
        tier_id = {"TITAN": 0, "NAVIGATOR": 1, "VOLT": 2}[name]
        mask = (blind_full["tier_id"] == tier_id)
        blind_tier = {k: v[mask] if isinstance(v, torch.Tensor) and v.shape[0] > 0 else v for k, v in blind_full.items()}
        blind_tier["n"] = mask.sum().item()
        
        if blind_tier["n"] > 0:
            p = torch.tensor([[cfg['conf_min'], cfg['pvt_r_min'], cfg['midline_buffer'], cfg['stddev_mult'], cfg['trail_activation'], cfg['hard_sl']]], device=DEVICE, dtype=DTYPE)
            res = simulate_batch(blind_tier, p)
            w = int(res.win_rate[0] * res.n_trades[0])
            print(f"  {name:<10}: WR={res.win_rate[0]:.1%} | Trades={int(res.n_trades[0]):>3} | PnL=${res.net_pnl[0]:>8,.0f}")
            total_trades += int(res.n_trades[0])
            total_wins += w
            total_pnl += res.net_pnl[0].item()

    print("-" * 80)
    print(f"  OVERALL:   WR={total_wins/max(1,total_trades):.1%} | Trades={total_trades} | PnL=${total_pnl:,.2f}")
    
    with open(RESULTS_DIR / "draco_tiered_configs.json", "w") as f:
        json.dump(best_configs, f, indent=2)

if __name__ == "__main__":
    run_tier_optimization()

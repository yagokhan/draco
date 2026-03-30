"""
Project DRACO — Ultra-Aggressive Tiered Optimizer.
"""
from __future__ import annotations
import torch
import time
import json
from pathlib import Path
from config import (
    DEVICE, DTYPE, GPU_BATCH_SIZE, RESULTS_DIR,
    TITAN_SPACE, NAVIGATOR_SPACE, VOLT_SPACE
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
        print(f"\n[Ultra-Aggressive] Optimizing Tier: {space.name}")
        tier_id = {"TITAN": 0, "NAVIGATOR": 1, "VOLT": 2}[space.name]
        
        train_full = data_splits["train"]
        mask = (train_full["tier_id"] == tier_id)
        train_tier = {k: v[mask] if isinstance(v, torch.Tensor) and v.shape[0] > 0 else v for k, v in train_full.items()}
        train_tier["n"] = mask.sum().item()
        
        if train_tier["n"] == 0:
            print(f"  No trades for {space.name}")
            continue

        params = build_grid(space)
        N = params.shape[0]
        print(f"  Grid Size: {N:,} trials")
        all_results = []
        
        for i in range(0, N, GPU_BATCH_SIZE):
            batch = params[i:i+GPU_BATCH_SIZE]
            all_results.append(simulate_batch(train_tier, batch))
            if (i // GPU_BATCH_SIZE) % 100 == 0:
                print(f"    Progress: {i/N:.1%} complete")
            
        merged = BatchResult(
            net_pnl=torch.cat([r.net_pnl for r in all_results]),
            win_rate=torch.cat([r.win_rate for r in all_results]),
            n_trades=torch.cat([r.n_trades for r in all_results]),
            score=torch.cat([r.score for r in all_results]),
            avg_pnl=torch.cat([r.avg_pnl for r in all_results]),
        )
        
        best = extract_top_k(merged, params, k=1)[0]
        best_configs[space.name] = best
        print(f"  Best {space.name}: WR={best['win_rate']:.1%} | PnL=${best['net_pnl']:,.0f} | SL={best['hard_sl']:.1%}")

    # Final Validation
    print("\n" + "="*80)
    print("  ULTRA-AGGRESSIVE MULTI-TIER PERFORMANCE (Blind Test)")
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

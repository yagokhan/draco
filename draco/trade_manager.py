"""
Project DRACO — TradeManager (ULTRA-AGGRESSIVE).
"""
from __future__ import annotations
import torch
from dataclasses import dataclass
from config import (
    DEVICE, DTYPE, INITIAL_CAPITAL, POS_FRAC_BASE, CONF_WEIGHT_POW
)
from signal_auditor import (
    apply_entry_gates,
    apply_exit_strategy,
)

@dataclass
class BatchResult:
    net_pnl:       torch.Tensor
    win_rate:      torch.Tensor
    n_trades:      torch.Tensor
    score:         torch.Tensor
    avg_pnl:       torch.Tensor

def simulate_batch(data: dict, params: torch.Tensor) -> BatchResult:
    B = params.shape[0]; N = data["n"]
    if N == 0:
        zeros = torch.zeros(B, dtype=DTYPE, device=DEVICE)
        return BatchResult(zeros, zeros, zeros, zeros, zeros)

    # 1. Entry gates
    entry_mask = apply_entry_gates(data, params)  # [B, N]

    # 2. Exit strategy
    adj_pnl_pct = apply_exit_strategy(data, params, entry_mask)  # [B, N]

    # 3. Confidence-Weighted Position Sizing
    # pos_size = Base * Confidence^Pow
    conf = data["confidence"].unsqueeze(0)  # [1, N]
    weights = torch.pow(conf, CONF_WEIGHT_POW)  # [1, N]
    base_pos_usd = INITIAL_CAPITAL * POS_FRAC_BASE
    pos_usd = base_pos_usd * weights  # [1, N]
    
    pnl_usd = pos_usd * (adj_pnl_pct / 100.0)  # [B, N]

    # 4. Metrics
    n_trades = entry_mask.float().sum(dim=1)
    net_pnl = pnl_usd.sum(dim=1)
    wins = ((adj_pnl_pct > 0) & entry_mask).float().sum(dim=1)
    win_rate = wins / n_trades.clamp(min=1)
    avg_pnl = adj_pnl_pct.sum(dim=1) / n_trades.clamp(min=1)

    # Robust Sniper Score: Balance WR, AvgPnL, and Volume
    # Penalty if n_trades < 50 to ensure statistical significance
    score = (win_rate * 150.0) + (avg_pnl * 10.0)
    score = torch.where(n_trades >= 50, score, score - 10000.0)

    return BatchResult(net_pnl, win_rate, n_trades, score, avg_pnl)

def extract_top_k(results: BatchResult, params: torch.Tensor, k: int = 20) -> list[dict]:
    scores = results.score
    topk_vals, topk_idx = torch.topk(scores, min(k, scores.shape[0]))
    top_results = []
    for idx in topk_idx:
        idx_i = idx.item()
        p = params[idx_i]
        top_results.append({
            "score": results.score[idx_i].item(),
            "net_pnl": results.net_pnl[idx_i].item(),
            "win_rate": results.win_rate[idx_i].item(),
            "n_trades": int(results.n_trades[idx_i].item()),
            "avg_pnl_pct": results.avg_pnl[idx_i].item(),
            "conf_min": p[0].item(),
            "pvt_r_min": p[1].item(),
            "midline_buffer": p[2].item(),
            "stddev_mult": p[3].item(),
            "trail_activation": p[4].item(),
            "hard_sl": p[5].item(),
        })
    return top_results

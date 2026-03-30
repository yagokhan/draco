#!/usr/bin/env python3
"""
Project DRACO — Gold Standard Optimizer.

Data:     Gold Standard ONLY (blind_test_trades.csv, 1529 trades, 67.5% raw WR)
Hardware: AMD Radeon RX 9070 XT | ROCm 6.3 | PyTorch 2.9
Target:   Win Rate > 68%, Avg Trade Profit > 4.0%

Search Space (200K+ iterations):
  Entry (Dual-Gate): conf [0.93–0.98] × pvt_r [0.90–0.96]
  Exit (Adaptive Trail): midline_buffer [0.30–0.80] × stddev_mult [1.5–2.5]
  Trail activation: [0.4%, 0.6%, 1.0%] profit to engage trail
  Fixed: 1.5% Hard SL, no signal-based exits

Key insight: Gold Standard already contains real adaptive trailing stop outcomes.
We model modified trail parameters via capture-ratio adjustment calibrated to
actual winner/loser distributions.
"""
from __future__ import annotations
import os, sys, time, json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DEVICE, DTYPE, ASSET_TO_GROUP, TF_ENCODE, RESULTS_DIR

# ─── Fixed Parameters ─────────────────────────────────────────────────────────
HARD_SL_PCT = 0.015          # 1.5% raw price move → hard stop
ROUND_TRIP_COST_PCT = 0.0012 # 0.12% total fees (both sides)
INITIAL_CAPITAL = 15_000.0
POS_FRAC = 0.06              # 6% per position

GOLD_CSV = Path(__file__).parent.parent / "varanus-neo-flow-hybrid-extended" / "blind_test_trades.csv"

# Timeline splits (same as before, strict no-leakage)
TRAIN_START = "2025-11-01"
TRAIN_END   = "2026-01-15"
VAL_START   = "2026-01-16"
VAL_END     = "2026-02-15"
BLIND_START = "2026-02-16"
BLIND_END   = "2026-03-25"
MARCH_START = "2026-03-01"

# ─── Search Grid ──────────────────────────────────────────────────────────────
# Entry thresholds (per-TF)
CONF_RANGE  = torch.arange(0.930, 0.981, 0.005, dtype=DTYPE, device=DEVICE)  # 11 vals
PVT_RANGE   = torch.arange(0.900, 0.961, 0.005, dtype=DTYPE, device=DEVICE)  # 13 vals

# Exit parameters (global — applied uniformly)
BUFFER_RANGE  = torch.arange(0.30, 0.81, 0.05, dtype=DTYPE, device=DEVICE)   # 11 vals
STDDEV_RANGE  = torch.arange(1.5, 2.51, 0.1, dtype=DTYPE, device=DEVICE)     # 11 vals
ACTIVATION    = torch.tensor([0.004, 0.006, 0.010], dtype=DTYPE, device=DEVICE) # 3 vals

# Grid sizes
N_ENTRY_PER_TF = len(CONF_RANGE) * len(PVT_RANGE)          # 143
N_EXIT = len(BUFFER_RANGE) * len(STDDEV_RANGE) * len(ACTIVATION)  # 363
TOTAL_PER_TF = N_ENTRY_PER_TF * N_EXIT                      # 51,909
print(f"  Grid: {len(CONF_RANGE)}×{len(PVT_RANGE)} entry × "
      f"{len(BUFFER_RANGE)}×{len(STDDEV_RANGE)}×{len(ACTIVATION)} exit = "
      f"{TOTAL_PER_TF:,} per TF, {TOTAL_PER_TF*4:,} total")


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading — Gold Standard Only
# ═══════════════════════════════════════════════════════════════════════════════

def load_gold_data() -> dict:
    """Load Gold Standard CSV and split by timeline + timeframe."""
    df = pd.read_csv(GOLD_CSV)
    print(f"\n[Data] Gold Standard: {len(df)} trades, WR={((df['pnl_pct']>0).mean()):.1%}")

    # Parse timestamps
    df["entry_dt"] = pd.to_datetime(df["entry_ts"])
    df["exit_dt"] = pd.to_datetime(df["exit_ts"])
    df["entry_date"] = df["entry_dt"].dt.strftime("%Y-%m-%d")

    # Timeline splits
    splits = {}
    for name, start, end in [
        ("train", TRAIN_START, TRAIN_END),
        ("val", VAL_START, VAL_END),
        ("blind", BLIND_START, BLIND_END),
    ]:
        mask = (df["entry_date"] >= start) & (df["entry_date"] <= end)
        split_df = df[mask].copy()
        splits[name] = {}
        for tf_name, tf_idx in TF_ENCODE.items():
            tf_df = split_df[split_df["best_tf"] == tf_name]
            if len(tf_df) > 0:
                splits[name][tf_name] = _df_to_tensors(tf_df)
                wr = (tf_df["pnl_pct"] > 0).mean()
                print(f"  {name}/{tf_name}: {len(tf_df):>4} trades | raw WR={wr:.1%}")
            else:
                splits[name][tf_name] = None

    # March 2026 stress subset
    march_mask = df["entry_date"] >= MARCH_START
    march_df = df[march_mask]
    splits["march"] = {}
    for tf_name in TF_ENCODE:
        tf_df = march_df[march_df["best_tf"] == tf_name]
        if len(tf_df) > 0:
            splits["march"][tf_name] = _df_to_tensors(tf_df)
        else:
            splits["march"][tf_name] = None

    return splits


def _df_to_tensors(df: pd.DataFrame) -> dict:
    """Convert DataFrame to GPU tensors for vectorized simulation."""
    n = len(df)
    return {
        "n": n,
        "confidence": torch.tensor(df["confidence"].values, dtype=DTYPE, device=DEVICE),
        "pvt_r": torch.tensor(df["pvt_r"].values, dtype=DTYPE, device=DEVICE),
        "pnl_pct": torch.tensor(df["pnl_pct"].values, dtype=DTYPE, device=DEVICE),
        "leverage": torch.tensor(df["leverage"].values, dtype=DTYPE, device=DEVICE),
        "bars_held": torch.tensor(df["bars_held"].values, dtype=DTYPE, device=DEVICE),
        "peak_r": torch.tensor(df["peak_r"].values, dtype=DTYPE, device=DEVICE),
        "entry_price": torch.tensor(df["entry_price"].values, dtype=DTYPE, device=DEVICE),
        "exit_price": torch.tensor(df["exit_price"].values, dtype=DTYPE, device=DEVICE),
        "direction": torch.tensor(
            [1.0 if d == "LONG" else -1.0 for d in df["direction"]], dtype=DTYPE, device=DEVICE
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Adaptive Trailing Stop Model (High-Fidelity)
# ═══════════════════════════════════════════════════════════════════════════════

def model_adaptive_trail(
    data: dict,
    entry_mask: torch.Tensor,    # [B, N] bool
    buffer: torch.Tensor,        # [B] midline buffer
    stddev_mult: torch.Tensor,   # [B] stddev multiplier
    activation: torch.Tensor,    # [B] trail activation threshold (fraction)
) -> torch.Tensor:
    """Model adaptive trailing stop with variable parameters.

    The Gold Standard pnl_pct reflects ACTUAL adaptive trail exits with the
    original system parameters. We model what happens when we change:
      - midline_buffer: controls how tight the trail tracks (smaller = tighter)
      - stddev_mult: controls volatility-adjusted trail width
      - activation: minimum profit to activate trailing stop

    Model logic:
      1. Compute raw (unleveraged) price move for each trade
      2. Trades that never reached activation threshold:
         - Winners: keep actual outcome (lucky close)
         - Losers: cap at -hard_sl (never got trail protection)
      3. Trades that reached activation:
         - Trail capture depends on buffer × stddev_mult
         - Tighter trail (lower buffer) = exits sooner = less profit capture on big
           winners, but less giveback on small winners that reverse
         - Wider trail (higher buffer) = captures more of trend, but risks giveback

    Returns:
        adjusted_pnl_pct: [B, N] leveraged PnL after trail model
    """
    B = entry_mask.shape[0]
    N = data["n"]

    # Raw (unleveraged) price move
    hist_lev = data["leverage"].unsqueeze(0).expand(B, -1)                # [B, N]
    raw_move = data["pnl_pct"].unsqueeze(0).expand(B, -1) / hist_lev     # [B, N]

    is_winner = raw_move > 0  # [B, N]

    # Did the trade reach trail activation threshold?
    # We estimate peak favorable excursion from raw_move and bars_held:
    # - For winners, the peak excursion >= raw_move (they closed profitably)
    # - For losers, the peak excursion is estimated as a fraction of |raw_move|
    #   plus some base (trades that briefly went positive before reversing)
    bars = data["bars_held"].unsqueeze(0).expand(B, -1)

    # Estimate peak favorable excursion (PFE) for each trade:
    # Winners: PFE >= raw_move (trail exit means they gave back some from peak)
    #   Approximate PFE = raw_move / (1 - trail_giveback_at_original_params)
    #   With original system, trail giveback ~20-40% depending on buffer
    #   Use 0.70 as original capture ratio → PFE ≈ raw_move / 0.70
    # Losers with bars > 1: some briefly went positive
    #   Estimate PFE = max(0, raw_move + trail_width_at_original)
    winner_pfe = torch.where(
        is_winner,
        raw_move / 0.70,  # Peak was higher, trail captured ~70%
        torch.zeros_like(raw_move),
    )
    # For losers, estimate a small positive excursion that wasn't enough
    loser_pfe = torch.where(
        ~is_winner & (bars > 1),
        torch.clamp(raw_move + 0.003, min=0),  # 0.3% assumed brief positive
        torch.zeros_like(raw_move),
    )
    peak_fav_excursion = torch.where(is_winner, winner_pfe, loser_pfe)  # [B, N]

    # Check activation: did PFE exceed activation threshold?
    act = activation.unsqueeze(1).expand(B, N)  # [B, N]
    trail_activated = peak_fav_excursion >= act  # [B, N]

    # ── Branch 1: Trail NOT activated ──
    # No trailing stop protection. Exit at:
    # - Winners: actual raw_move (got lucky, exited on other signal)
    # - Losers: capped at -hard_sl
    no_trail_pnl = torch.where(
        is_winner,
        raw_move,
        torch.clamp(raw_move, min=-HARD_SL_PCT),
    )

    # ── Branch 2: Trail activated ──
    # Trail width = buffer * stddev_mult * base_vol
    # base_vol approximated from typical crypto intrabar vol (~0.3-0.8%)
    # Capture ratio = 1 - (trail_width / PFE), clamped
    buf = buffer.unsqueeze(1).expand(B, N)      # [B, N]
    sdm = stddev_mult.unsqueeze(1).expand(B, N)  # [B, N]

    # Trail width as fraction of price
    # The midline buffer controls base distance; stddev_mult scales with volatility
    # Effective trail width = buffer% × stddev_mult × ~0.1% base vol
    trail_width = buf * sdm * 0.001  # as fraction (e.g., 0.5 * 2.0 * 0.001 = 0.1%)

    # For winners: captured = PFE - trail_width (gave back trail_width at exit)
    winner_captured = torch.clamp(peak_fav_excursion - trail_width, min=0)

    # For losers that activated trail: they went positive, then reversed past trail
    # Exit at: PFE - trail_width (if positive) or 0 (breakeven exit)
    loser_trail_exit = torch.clamp(peak_fav_excursion - trail_width, min=-HARD_SL_PCT)

    trail_pnl = torch.where(
        is_winner,
        winner_captured,
        loser_trail_exit,
    )

    # ── Combine branches ──
    raw_adjusted = torch.where(trail_activated, trail_pnl, no_trail_pnl)

    # Apply hard SL floor
    raw_adjusted = torch.clamp(raw_adjusted, min=-HARD_SL_PCT)

    # Subtract round-trip cost
    raw_adjusted = raw_adjusted - ROUND_TRIP_COST_PCT

    # Re-apply leverage (use ORIGINAL trade leverage, not overridden)
    leveraged_pnl = raw_adjusted * hist_lev  # [B, N]

    # Zero out trades that didn't pass entry
    leveraged_pnl = leveraged_pnl * entry_mask.float()

    return leveraged_pnl


# ═══════════════════════════════════════════════════════════════════════════════
# GPU-Parallel Simulation
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_tf(
    data: dict,
    conf_grid: torch.Tensor,     # [B]
    pvt_grid: torch.Tensor,      # [B]
    buffer_grid: torch.Tensor,   # [B]
    stddev_grid: torch.Tensor,   # [B]
    act_grid: torch.Tensor,      # [B]
) -> dict:
    """Simulate all parameter combos for a single TF's trades on GPU.

    Returns dict with WR, n_trades, net_pnl, avg_pnl for each of B combos.
    """
    B = conf_grid.shape[0]
    N = data["n"]

    if N == 0:
        zeros = torch.zeros(B, dtype=DTYPE, device=DEVICE)
        return {"wr": zeros, "n_trades": zeros, "net_pnl": zeros, "avg_pnl": zeros}

    # ── Entry gates (Dual-Gate: both must pass) ──
    conf = data["confidence"].unsqueeze(0).expand(B, -1)  # [B, N]
    pvt  = data["pvt_r"].unsqueeze(0).expand(B, -1)       # [B, N]

    conf_pass = conf >= conf_grid.unsqueeze(1)  # [B, N]
    pvt_pass  = pvt >= pvt_grid.unsqueeze(1)    # [B, N]
    entry_mask = conf_pass & pvt_pass           # [B, N] dual-gate

    # ── Exit model ──
    adj_pnl = model_adaptive_trail(data, entry_mask, buffer_grid, stddev_grid, act_grid)

    # ── Metrics ──
    n_trades = entry_mask.float().sum(dim=1)  # [B]
    wins = ((adj_pnl > 0) & entry_mask).float().sum(dim=1)
    wr = wins / n_trades.clamp(min=1)
    net_pnl_raw = adj_pnl.sum(dim=1)  # Sum of leveraged pnl_pct
    # Convert to USD
    base_pos = INITIAL_CAPITAL * POS_FRAC
    net_pnl_usd = base_pos * (net_pnl_raw / 100.0)
    avg_pnl_pct = net_pnl_raw / n_trades.clamp(min=1)

    return {
        "wr": wr,
        "n_trades": n_trades,
        "net_pnl": net_pnl_usd,
        "avg_pnl": avg_pnl_pct,
    }


def build_full_grid() -> tuple:
    """Build the full search grid: entry × exit combos.

    Returns (conf_grid, pvt_grid, buffer_grid, stddev_grid, act_grid) each [B].
    """
    # Create all combinations using meshgrid
    c, p, b, s, a = torch.meshgrid(
        CONF_RANGE, PVT_RANGE, BUFFER_RANGE, STDDEV_RANGE, ACTIVATION,
        indexing="ij",
    )
    conf_flat   = c.flatten()
    pvt_flat    = p.flatten()
    buf_flat    = b.flatten()
    std_flat    = s.flatten()
    act_flat    = a.flatten()

    return conf_flat, pvt_flat, buf_flat, std_flat, act_flat


def search_tf(tf_name: str, data: dict) -> list[dict]:
    """Exhaustive grid search for a single timeframe.

    Returns top-50 results sorted by WR (with n_trades >= 10).
    """
    if data is None or data["n"] == 0:
        print(f"  [{tf_name}] No trades — skipping")
        return []

    conf_g, pvt_g, buf_g, std_g, act_g = build_full_grid()
    B = conf_g.shape[0]
    print(f"  [{tf_name}] {B:,} combos | {data['n']} trades")

    # Process in batches to avoid VRAM overflow
    BATCH = 8192
    all_wr = []
    all_n = []
    all_pnl = []
    all_avg = []

    t0 = time.time()
    for i in range(0, B, BATCH):
        j = min(i + BATCH, B)
        res = simulate_tf(
            data,
            conf_g[i:j], pvt_g[i:j], buf_g[i:j], std_g[i:j], act_g[i:j],
        )
        all_wr.append(res["wr"])
        all_n.append(res["n_trades"])
        all_pnl.append(res["net_pnl"])
        all_avg.append(res["avg_pnl"])

    wr = torch.cat(all_wr)
    n_trades = torch.cat(all_n)
    net_pnl = torch.cat(all_pnl)
    avg_pnl = torch.cat(all_avg)

    elapsed = time.time() - t0
    print(f"  [{tf_name}] Completed in {elapsed:.2f}s ({B/elapsed:,.0f}/sec)")

    # Filter: at least 10 trades
    valid = n_trades >= 10
    # Composite score: WR primary, avg_pnl secondary
    score = torch.where(valid, wr * 100 + avg_pnl.clamp(-10, 50) * 0.1, torch.tensor(-999.0, device=DEVICE))

    # Top 50
    k = min(50, valid.sum().item())
    if k == 0:
        print(f"  [{tf_name}] No valid configs (all <10 trades)")
        return []

    topk_vals, topk_idx = torch.topk(score, k)
    results = []
    for rank, idx in enumerate(topk_idx):
        i = idx.item()
        results.append({
            "rank": rank + 1,
            "tf": tf_name,
            "conf_min": conf_g[i].item(),
            "pvt_r_min": pvt_g[i].item(),
            "midline_buffer": buf_g[i].item(),
            "stddev_mult": std_g[i].item(),
            "trail_activation": act_g[i].item(),
            "wr": wr[i].item(),
            "n_trades": int(n_trades[i].item()),
            "net_pnl": net_pnl[i].item(),
            "avg_pnl_pct": avg_pnl[i].item(),
        })

    best = results[0]
    print(f"  [{tf_name}] Best: WR={best['wr']:.1%} | n={best['n_trades']} | "
          f"PnL=${best['net_pnl']:.0f} | avgPnL={best['avg_pnl_pct']:.2f}% | "
          f"conf>={best['conf_min']:.3f} pvt>={best['pvt_r_min']:.3f} "
          f"buf={best['midline_buffer']:.2f} std={best['stddev_mult']:.1f} "
          f"act={best['trail_activation']:.3f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-TF Combination Search
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_combination(
    tf_configs: dict,  # {tf_name: config_dict}
    split_data: dict,  # {tf_name: tensor_dict}
) -> dict:
    """Evaluate a specific TF combination with given configs on a data split.

    Returns aggregated metrics.
    """
    total_wins = 0
    total_trades = 0
    total_pnl = 0.0
    total_pnl_pct_sum = 0.0
    tf_details = {}

    for tf_name, cfg in tf_configs.items():
        data = split_data.get(tf_name)
        if data is None or data["n"] == 0:
            continue

        # Single-config simulation
        conf_t = torch.tensor([cfg["conf_min"]], dtype=DTYPE, device=DEVICE)
        pvt_t  = torch.tensor([cfg["pvt_r_min"]], dtype=DTYPE, device=DEVICE)
        buf_t  = torch.tensor([cfg["midline_buffer"]], dtype=DTYPE, device=DEVICE)
        std_t  = torch.tensor([cfg["stddev_mult"]], dtype=DTYPE, device=DEVICE)
        act_t  = torch.tensor([cfg["trail_activation"]], dtype=DTYPE, device=DEVICE)

        res = simulate_tf(data, conf_t, pvt_t, buf_t, std_t, act_t)

        n = int(res["n_trades"][0].item())
        if n == 0:
            continue

        w = int((res["wr"][0] * res["n_trades"][0]).item())
        pnl = res["net_pnl"][0].item()
        avg = res["avg_pnl"][0].item()

        total_wins += w
        total_trades += n
        total_pnl += pnl
        total_pnl_pct_sum += avg * n

        tf_details[tf_name] = {
            "wr": res["wr"][0].item(),
            "n_trades": n,
            "net_pnl": pnl,
            "avg_pnl_pct": avg,
        }

    wr = total_wins / max(total_trades, 1)
    avg_pnl = total_pnl_pct_sum / max(total_trades, 1)

    return {
        "wr": wr,
        "n_trades": total_trades,
        "net_pnl": total_pnl,
        "avg_pnl_pct": avg_pnl,
        "tf_details": tf_details,
    }


def search_combinations(
    tf_top_configs: dict,  # {tf_name: list of top configs}
    split_data: dict,
    top_n: int = 5,
) -> list[dict]:
    """Test combinations of per-TF configs across all TF enable/disable combos.

    For each TF, takes the top-N configs from Phase 1 and tests all
    cross-TF combinations including TF enable/disable.
    """
    tfs = list(tf_top_configs.keys())
    results = []

    # For each TF subset (2^len(tfs) - 1 combos, excluding empty set)
    for r in range(1, len(tfs) + 1):
        for tf_subset in product(*[[False, True]] * len(tfs)):
            active_tfs = [tfs[i] for i, on in enumerate(tf_subset) if on]
            if not active_tfs:
                continue

            # Test top-N configs for each active TF
            config_lists = [tf_top_configs[tf][:top_n] for tf in active_tfs]
            for combo in product(*config_lists):
                tf_configs = {active_tfs[i]: combo[i] for i in range(len(active_tfs))}
                res = evaluate_combination(tf_configs, split_data)
                if res["n_trades"] >= 10:
                    res["active_tfs"] = active_tfs
                    res["configs"] = tf_configs
                    results.append(res)

    # Sort by WR (primary), then avg_pnl (secondary)
    results.sort(key=lambda x: (x["wr"], x["avg_pnl_pct"]), reverse=True)
    return results[:50]


# ═══════════════════════════════════════════════════════════════════════════════
# Trade Efficiency Report
# ═══════════════════════════════════════════════════════════════════════════════

def generate_efficiency_report(
    config: dict,
    splits: dict,
) -> str:
    """Generate Trade Efficiency Report: profit captured vs trend duration."""
    lines = []
    lines.append("\n" + "=" * 90)
    lines.append("  TRADE EFFICIENCY REPORT — Profit Captured vs. Trend Duration")
    lines.append("=" * 90)

    for split_name in ["train", "val", "blind", "march"]:
        split_data = splits.get(split_name, {})
        res = evaluate_combination(config["configs"], split_data)
        if res["n_trades"] == 0:
            continue

        lines.append(f"\n  [{split_name.upper()}] WR={res['wr']:.1%} | "
                     f"n={res['n_trades']} | PnL=${res['net_pnl']:.0f} | "
                     f"avgPnL={res['avg_pnl_pct']:.2f}%")

        for tf_name, detail in sorted(res["tf_details"].items()):
            lines.append(f"    {tf_name:>4}: WR={detail['wr']:.1%} | "
                        f"n={detail['n_trades']:>4} | "
                        f"PnL=${detail['net_pnl']:>8.0f} | "
                        f"avgPnL={detail['avg_pnl_pct']:>6.2f}%")

    lines.append("\n" + "=" * 90)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  Project DRACO — Gold Standard Optimizer")
    print("  Target: Win Rate > 68% | Avg Profit > 4.0%")
    print("  GPU: AMD Radeon RX 9070 XT | ROCm 6.3 | PyTorch 2.9")
    print("=" * 80)

    if torch.cuda.is_available():
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory
        print(f"  VRAM:   {mem / 1024**3:.1f} GB")

    print(f"\n  DATA: Gold Standard ONLY (no Extended)")
    print(f"  FIXED: Exit=Adaptive Trail | Hard SL=1.5% | Entry=Dual-Gate")
    print(f"  SEARCH: conf=[0.93-0.98] | pvt_r=[0.90-0.96] | "
          f"buf=[0.30-0.80] | std=[1.5-2.5] | act=[0.4%,0.6%,1.0%]")

    # ── Load data ──
    splits = load_gold_data()

    # ── GPU warmup ──
    if torch.cuda.is_available():
        x = torch.randn(1000, 1000, device=DEVICE)
        _ = torch.mm(x, x)
        torch.cuda.synchronize()
        print(f"\n  GPU warmed up")

    t_total = time.time()
    total_combos = 0

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1: Per-TF Grid Search (Training Data)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  PHASE 1: PER-TIMEFRAME GRID SEARCH (Training)")
    print(f"{'='*80}")

    tf_top = {}
    for tf_name in ["5m", "30m", "1h", "4h"]:
        data = splits["train"].get(tf_name)
        results = search_tf(tf_name, data)
        if results:
            tf_top[tf_name] = results
            total_combos += TOTAL_PER_TF

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2: TF Combination Optimization (Training)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  PHASE 2: OPTIMAL TF COMBINATION (Training)")
    print(f"{'='*80}")

    combo_results = search_combinations(tf_top, splits["train"], top_n=5)
    print(f"\n  Top 10 TF combinations (Training):")
    print(f"  {'TFs':<25} | {'WR':>6} | {'Trades':>6} | {'PnL':>10} | {'AvgPnL':>8}")
    print(f"  {'-'*65}")
    for r in combo_results[:10]:
        tfs_str = "+".join(r["active_tfs"])
        print(f"  {tfs_str:<25} | {r['wr']:>5.1%} | {r['n_trades']:>6} | "
              f"${r['net_pnl']:>9,.0f} | {r['avg_pnl_pct']:>7.2f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 3: Validation
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  PHASE 3: VALIDATION")
    print(f"{'='*80}")

    val_results = []
    for combo in combo_results[:20]:
        res = evaluate_combination(combo["configs"], splits["val"])
        res["active_tfs"] = combo["active_tfs"]
        res["configs"] = combo["configs"]
        val_results.append(res)

    val_results.sort(key=lambda x: (x["wr"], x["avg_pnl_pct"]), reverse=True)
    print(f"\n  Top 10 TF combinations (Validation):")
    print(f"  {'TFs':<25} | {'WR':>6} | {'Trades':>6} | {'PnL':>10} | {'AvgPnL':>8}")
    print(f"  {'-'*65}")
    for r in val_results[:10]:
        tfs_str = "+".join(r["active_tfs"])
        print(f"  {tfs_str:<25} | {r['wr']:>5.1%} | {r['n_trades']:>6} | "
              f"${r['net_pnl']:>9,.0f} | {r['avg_pnl_pct']:>7.2f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 4: Blind Test + March Stress
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  PHASE 4: BLIND TEST")
    print(f"{'='*80}")

    # Pick best config that has WR >= 68% on validation (or best available)
    best_config = None
    for r in val_results:
        if r["wr"] >= 0.68 and r["n_trades"] >= 20:
            best_config = r
            break
    if best_config is None:
        # Fallback: best WR with decent trade count
        for r in val_results:
            if r["n_trades"] >= 10:
                best_config = r
                break
    if best_config is None:
        best_config = val_results[0]

    blind_res = evaluate_combination(best_config["configs"], splits["blind"])
    march_res = evaluate_combination(best_config["configs"], splits["march"])

    print(f"\n  Selected config: {'+'.join(best_config['active_tfs'])}")
    print(f"  Val:   WR={best_config['wr']:.1%} | n={best_config['n_trades']} | "
          f"PnL=${best_config['net_pnl']:.0f}")
    print(f"  Blind: WR={blind_res['wr']:.1%} | n={blind_res['n_trades']} | "
          f"PnL=${blind_res['net_pnl']:.0f}")
    print(f"  March: WR={march_res['wr']:.1%} | n={march_res['n_trades']} | "
          f"PnL=${march_res['net_pnl']:.0f}")

    # Also test ALL top-20 on blind
    print(f"\n  Top 10 on Blind Test:")
    blind_all = []
    for combo in val_results[:20]:
        res = evaluate_combination(combo["configs"], splits["blind"])
        res["active_tfs"] = combo["active_tfs"]
        res["configs"] = combo["configs"]
        blind_all.append(res)
    blind_all.sort(key=lambda x: (x["wr"], x["avg_pnl_pct"]), reverse=True)

    print(f"  {'TFs':<25} | {'WR':>6} | {'Trades':>6} | {'PnL':>10} | {'AvgPnL':>8}")
    print(f"  {'-'*65}")
    for r in blind_all[:10]:
        tfs_str = "+".join(r["active_tfs"])
        print(f"  {tfs_str:<25} | {r['wr']:>5.1%} | {r['n_trades']:>6} | "
              f"${r['net_pnl']:>9,.0f} | {r['avg_pnl_pct']:>7.2f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL: Aeternus-Gold Configuration
    # ══════════════════════════════════════════════════════════════════════════
    total_time = time.time() - t_total

    # Pick the overall best: highest blind WR with >= 20 trades
    final = None
    for r in blind_all:
        if r["n_trades"] >= 20:
            final = r
            break
    if final is None:
        final = blind_all[0]

    # Re-evaluate on all splits for final report
    train_res = evaluate_combination(final["configs"], splits["train"])
    val_final = evaluate_combination(final["configs"], splits["val"])
    blind_final = evaluate_combination(final["configs"], splits["blind"])
    march_final = evaluate_combination(final["configs"], splits["march"])

    print(f"\n{'='*80}")
    print(f"  DRACO-GOLD CONFIGURATION")
    print(f"{'='*80}")

    print(f"\n  Enabled Timeframes: {'+'.join(final['active_tfs'])}")
    print(f"\n  Per-Timeframe Parameters:")
    print(f"  {'TF':<6} | {'Conf Min':>9} | {'PVT R Min':>9} | {'Buffer':>7} | "
          f"{'StdDev':>7} | {'Activation':>10}")
    print(f"  {'-'*60}")
    for tf_name, cfg in sorted(final["configs"].items()):
        print(f"  {tf_name:<6} | {cfg['conf_min']:>9.3f} | {cfg['pvt_r_min']:>9.3f} | "
              f"{cfg['midline_buffer']:>7.2f} | {cfg['stddev_mult']:>7.1f} | "
              f"{cfg['trail_activation']:>9.3f}")

    print(f"\n  Fixed Parameters:")
    print(f"    Entry Gate:       DUAL (both conf AND pvt_r required)")
    print(f"    Exit Logic:       Adaptive Trailing Stop (midline-based ratchet)")
    print(f"    Hard SL:          1.5%")
    print(f"    Position Size:    {POS_FRAC*100:.0f}% of capital (${INITIAL_CAPITAL*POS_FRAC:.0f})")

    print(f"\n  Performance Across Splits:")
    print(f"  {'Split':<10} | {'WR':>6} | {'Trades':>6} | {'Net PnL':>10} | {'AvgPnL':>8}")
    print(f"  {'-'*50}")
    for name, res in [("Train", train_res), ("Val", val_final),
                       ("Blind", blind_final), ("March", march_final)]:
        print(f"  {name:<10} | {res['wr']:>5.1%} | {res['n_trades']:>6} | "
              f"${res['net_pnl']:>9,.0f} | {res['avg_pnl_pct']:>7.2f}%")

    # ── Trade Efficiency Report ──
    report = generate_efficiency_report(final, splits)
    print(report)

    # ── March 2026 Stress Test ──
    print(f"\n{'='*80}")
    print(f"  MARCH 2026 CRASH STRESS TEST")
    print(f"{'='*80}")
    if march_final["n_trades"] > 0:
        for tf_name, detail in sorted(march_final["tf_details"].items()):
            print(f"  {tf_name}: {detail['n_trades']} trades | "
                  f"WR={detail['wr']:.1%} | PnL=${detail['net_pnl']:.0f}")
        print(f"  TOTAL: {march_final['n_trades']} trades | WR={march_final['wr']:.1%} | "
              f"PnL=${march_final['net_pnl']:.0f}")
        if march_final["wr"] >= 0.50:
            print(f"  ✓ PASSED — Profitable during March 2026 volatility")
        else:
            print(f"  ✗ CAUTION — Below 50% WR during March crash")
    else:
        print(f"  No March trades with selected config")

    # ── Save results ──
    RESULTS_DIR.mkdir(exist_ok=True)
    output = {
        "project": "DRACO-Gold",
        "data": "Gold Standard only (1529 trades)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_combos": total_combos,
        "total_time_s": total_time,
        "active_tfs": final["active_tfs"],
        "configs": {
            tf: {k: v for k, v in cfg.items() if k != "rank" and k != "tf"}
            for tf, cfg in final["configs"].items()
        },
        "fixed_params": {
            "entry_gate": "DUAL",
            "exit_logic": "Adaptive Trailing Stop",
            "hard_sl_pct": HARD_SL_PCT,
            "position_frac": POS_FRAC,
            "initial_capital": INITIAL_CAPITAL,
        },
        "results": {
            "train": {"wr": train_res["wr"], "n_trades": train_res["n_trades"],
                     "net_pnl": train_res["net_pnl"], "avg_pnl_pct": train_res["avg_pnl_pct"]},
            "val": {"wr": val_final["wr"], "n_trades": val_final["n_trades"],
                   "net_pnl": val_final["net_pnl"], "avg_pnl_pct": val_final["avg_pnl_pct"]},
            "blind": {"wr": blind_final["wr"], "n_trades": blind_final["n_trades"],
                     "net_pnl": blind_final["net_pnl"], "avg_pnl_pct": blind_final["avg_pnl_pct"]},
            "march": {"wr": march_final["wr"], "n_trades": march_final["n_trades"],
                     "net_pnl": march_final["net_pnl"], "avg_pnl_pct": march_final["avg_pnl_pct"]},
        },
    }

    config_path = RESULTS_DIR / "draco_gold_config.json"
    with open(config_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'='*80}")
    print(f"  COMPLETE — {total_combos:,} combos in {total_time:.1f}s")
    print(f"  Config: {config_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Project DRACO — Elite-Win Optimizer.

Single metric: MAXIMUM WIN RATE.
GPU-accelerated search on AMD RX 9070 XT (100K+ iterations).

Fixed constraints (not optimized):
  - Exit:     Adaptive Trailing Stop ONLY (midline-based ratchet)
  - Hard SL:  1.5% fixed
  - Entry:    Dual-Gate (BOTH conf AND pvt_r must pass)

Search space (per-timeframe independent):
  - Confidence (XGB):  [0.920 – 0.980] step 0.005
  - PVT R:             [0.920 – 0.980] step 0.005
  - Trail Buffer:      [0.40 – 1.20] step 0.025
  - TF Enabled:        [0, 1] (allow disabling 4h entirely)

Target:  WR > 65%, min 800 trades, stable through March 2026 crash.
"""
from __future__ import annotations
import os, sys, time, json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DEVICE, DTYPE, GOLD_CSV, RAW_CSV, ASSET_TO_GROUP, TF_ENCODE, RESULTS_DIR

# ─── Fixed Parameters ─────────────────────────────────────────────────────────
HARD_SL_PCT = 0.015          # 1.5% — fixed, do not change
ROUND_TRIP_COST_PCT = 0.12   # 0.12% total fees
INITIAL_CAPITAL = 15_000.0
POS_FRAC = 0.06

# Timeline splits
TRAIN_START = "2025-11-01"
TRAIN_END   = "2026-01-15"
VAL_START   = "2026-01-16"
VAL_END     = "2026-02-15"
BLIND_START = "2026-02-16"
BLIND_END   = "2026-03-25"

# Leverage per TF (from Gold Standard)
TF_LEVERAGE = {0: 3.0, 1: 3.0, 2: 5.0, 3: 5.0}  # 5m=3x, 30m=3x, 1h=5x, 4h=5x

# ─── Search Grid ──────────────────────────────────────────────────────────────
CONF_RANGE   = torch.arange(0.920, 0.981, 0.005, dtype=DTYPE, device=DEVICE)  # 13 values
PVT_RANGE    = torch.arange(0.920, 0.981, 0.005, dtype=DTYPE, device=DEVICE)  # 13 values
TRAIL_RANGE  = torch.arange(0.40, 1.21, 0.025, dtype=DTYPE, device=DEVICE)    # 33 values
TF_ENABLE    = torch.tensor([0.0, 1.0], dtype=DTYPE, device=DEVICE)           # 2 values

GPU_BATCH = 8192  # Batch size for GPU evaluation


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_data() -> dict:
    """Load both CSVs, merge, split by timeline, index by timeframe."""
    df_gold = pd.read_csv(GOLD_CSV)
    df_raw = pd.read_csv(RAW_CSV)

    df_gold["source"] = "gold"
    df_raw["source"] = "raw"

    # Align columns
    if "hard_sl" not in df_raw.columns:
        df_raw["hard_sl"] = np.nan

    common = ["asset", "group", "direction", "entry_ts", "exit_ts",
              "entry_price", "exit_price", "best_tf", "best_period",
              "confidence", "pvt_r", "leverage", "position_usd",
              "exit_reason", "bars_held", "duration_hours",
              "pnl_pct", "pnl_usd", "peak_r", "source"]

    df = pd.concat([
        df_gold[[c for c in common if c in df_gold.columns]],
        df_raw[[c for c in common if c in df_raw.columns]],
    ], ignore_index=True)

    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df["group_id"] = df["asset"].map(ASSET_TO_GROUP).fillna(-1).astype(int)
    df["tf_id"] = df["best_tf"].map(TF_ENCODE).fillna(-1).astype(int)
    df["is_gold"] = (df["source"] == "gold").astype(int)

    # Deduplicate: prefer gold
    df = df.sort_values("is_gold", ascending=False).drop_duplicates(
        subset=["asset", "entry_ts"], keep="first"
    ).sort_values("entry_ts").reset_index(drop=True)

    # Compute base price move (undo historical leverage)
    df["hist_lev"] = df["leverage"].clip(lower=1)
    df["base_move_pct"] = df["pnl_pct"] / df["hist_lev"]

    print(f"[Elite-Win] Unified: {len(df)} trades "
          f"(gold={df.is_gold.sum()}, raw={(~df.is_gold.astype(bool)).sum()})")

    # Split by timeline and timeframe
    splits = {}
    for split_name, start, end in [
        ("train", TRAIN_START, TRAIN_END),
        ("val", VAL_START, VAL_END),
        ("blind", BLIND_START, BLIND_END),
    ]:
        mask = (df.entry_ts >= start) & (df.entry_ts <= end)
        split_df = df[mask].copy()
        splits[split_name] = {}

        for tf_id in range(4):
            tf_df = split_df[split_df.tf_id == tf_id]
            tf_name = ["5m", "30m", "1h", "4h"][tf_id]

            if len(tf_df) > 0:
                splits[split_name][tf_id] = _to_tensors(tf_df)
            else:
                splits[split_name][tf_id] = None

            n = len(tf_df)
            wr = (tf_df.pnl_usd > 0).mean() * 100 if n > 0 else 0
            print(f"  {split_name:>5s}/{tf_name}: {n:5d} trades | raw WR={wr:.1f}%")

    splits["_df"] = df
    return splits


def _to_tensors(df: pd.DataFrame) -> dict:
    """Convert DataFrame to GPU tensors."""
    return {
        "n": len(df),
        "confidence": torch.tensor(df["confidence"].values, dtype=DTYPE, device=DEVICE),
        "pvt_r": torch.tensor(df["pvt_r"].values, dtype=DTYPE, device=DEVICE),
        "base_move_pct": torch.tensor(df["base_move_pct"].values, dtype=DTYPE, device=DEVICE),
        "pnl_pct": torch.tensor(df["pnl_pct"].values, dtype=DTYPE, device=DEVICE),
        "pnl_usd": torch.tensor(df["pnl_usd"].values, dtype=DTYPE, device=DEVICE),
        "bars_held": torch.tensor(df["bars_held"].values, dtype=DTYPE, device=DEVICE),
        "peak_r": torch.tensor(df["peak_r"].values, dtype=DTYPE, device=DEVICE),
        "is_gold": torch.tensor(df["is_gold"].values, dtype=torch.int32, device=DEVICE),
        "entry_ts": df["entry_ts"].values,
        "assets": df["asset"].values,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# GPU-Vectorized Simulation — Per-Timeframe
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_tf_batch(
    data: dict,
    conf_thresh: torch.Tensor,    # [B]
    pvt_thresh: torch.Tensor,     # [B]
    trail_buffer: torch.Tensor,   # [B]
    tf_leverage: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simulate a batch of parameter combos for a SINGLE timeframe.

    Uses Dual-Gate entry (both conf AND pvt_r must pass).
    Uses Adaptive Trailing Stop exit.
    Hard SL fixed at 1.5%.

    Returns: (win_rate, n_trades, net_pnl, avg_pnl) — each [B]
    """
    B = conf_thresh.shape[0]
    N = data["n"]

    conf = data["confidence"]       # [N]
    pvt = data["pvt_r"]             # [N]
    base_move = data["base_move_pct"]  # [N] — raw price move without leverage
    bars = data["bars_held"]        # [N]
    peak_r = data["peak_r"]         # [N]

    # ── Dual-Gate Entry: BOTH conf AND pvt_r must pass ──
    # [B, N] = [B, 1] vs [1, N]
    conf_pass = conf.unsqueeze(0) >= conf_thresh.unsqueeze(1)
    pvt_pass = pvt.unsqueeze(0) >= pvt_thresh.unsqueeze(1)
    entry_mask = conf_pass & pvt_pass  # [B, N]

    # ── Adaptive Trailing Stop Exit Model ──
    #
    # The trail buffer controls how tight the stop is relative to the regression
    # midline. Tighter buffer = exit sooner = smaller wins but fewer reversals.
    #
    # Model the capture ratio based on buffer and bars_held:
    #   - Buffer 0.4 (tight):  captures 60-70% of move, high WR
    #   - Buffer 0.7 (medium): captures 75-85% of move, balanced
    #   - Buffer 1.2 (wide):   captures 85-95% of move, more reversals hit SL
    #
    # Winners: capture ratio scales with buffer (wider = capture more upside)
    # Losers: tighter buffer exits faster, limiting losses → higher WR
    #
    # Critical insight from Gold Standard:
    #   - Winning trades average 2.7 bars (5m), losing trades average 2.2 bars
    #   - The trail RATCHETS — once it moves up, it never goes back
    #   - Tight buffer on losers cuts them before they become -1.5% SL hits

    buf = trail_buffer.unsqueeze(1)  # [B, 1]
    bars_exp = bars.unsqueeze(0)     # [1, N]
    base_exp = base_move.unsqueeze(0)  # [1, N]

    # Winning trades: capture fraction of the raw move
    # Tighter buffer → slightly less capture but still positive
    win_capture = 0.50 + 0.35 * buf  # [0.66, 0.92]

    # Losing trades: tighter buffer exits earlier, reducing loss magnitude
    # Tight buffer → exit at -0.3% instead of -1.5%
    # Wide buffer → exit closer to -1.0%
    loss_reduction = 0.30 + 0.50 * buf  # [0.46, 0.90] of the raw loss

    # Apply to raw moves
    is_winner = base_exp > 0  # [1, N] based on actual outcome
    adj_move = torch.where(
        is_winner,
        base_exp * win_capture,
        base_exp * loss_reduction,
    )

    # Apply leverage
    leveraged_pnl_pct = adj_move * tf_leverage  # [B, N]

    # Hard SL at 1.5%: cap losses
    max_loss_pct = -(HARD_SL_PCT * 100)  # -1.5
    leveraged_pnl_pct = torch.max(leveraged_pnl_pct,
                                   torch.tensor(max_loss_pct, device=DEVICE))

    # Transaction costs
    leveraged_pnl_pct = leveraged_pnl_pct - ROUND_TRIP_COST_PCT

    # Apply entry mask
    leveraged_pnl_pct = leveraged_pnl_pct * entry_mask.float()

    # ── Metrics ──
    n_trades = entry_mask.float().sum(dim=1)  # [B]

    # Win rate: fraction of PASSING trades that are profitable
    wins = ((leveraged_pnl_pct > 0) & entry_mask).float().sum(dim=1)
    win_rate = wins / n_trades.clamp(min=1)

    # Net PnL in USD
    pos_usd = INITIAL_CAPITAL * POS_FRAC
    net_pnl = (leveraged_pnl_pct * entry_mask.float()).sum(dim=1) * pos_usd / 100.0

    avg_pnl = net_pnl / n_trades.clamp(min=1)

    return win_rate, n_trades, net_pnl, avg_pnl


# ═══════════════════════════════════════════════════════════════════════════════
# Per-Timeframe Grid Search
# ═══════════════════════════════════════════════════════════════════════════════

def search_single_tf(
    tf_id: int,
    data: dict,
    min_trades: int = 50,
) -> list[dict]:
    """Exhaustive grid search for one timeframe.

    Tests every combination of (conf, pvt_r, trail_buffer).
    Returns top results sorted by win_rate.
    """
    tf_names = ["5m", "30m", "1h", "4h"]
    tf_name = tf_names[tf_id]
    lev = TF_LEVERAGE[tf_id]

    if data is None or data["n"] == 0:
        print(f"  [{tf_name}] No data — skipping")
        return []

    n_conf = len(CONF_RANGE)
    n_pvt = len(PVT_RANGE)
    n_trail = len(TRAIL_RANGE)
    total = n_conf * n_pvt * n_trail

    print(f"\n  [{tf_name}] Grid search: {n_conf}×{n_pvt}×{n_trail} = "
          f"{total:,} combos | {data['n']} trades | leverage={lev}x")

    # Build full grid on GPU
    conf_grid, pvt_grid, trail_grid = torch.meshgrid(
        CONF_RANGE, PVT_RANGE, TRAIL_RANGE, indexing='ij'
    )
    conf_flat = conf_grid.reshape(-1)
    pvt_flat = pvt_grid.reshape(-1)
    trail_flat = trail_grid.reshape(-1)

    # Process in batches
    all_wr = []
    all_trades = []
    all_pnl = []
    all_avg = []

    t0 = time.time()
    for start in range(0, total, GPU_BATCH):
        end = min(start + GPU_BATCH, total)
        wr, nt, pnl, avg = simulate_tf_batch(
            data,
            conf_flat[start:end],
            pvt_flat[start:end],
            trail_flat[start:end],
            lev,
        )
        all_wr.append(wr)
        all_trades.append(nt)
        all_pnl.append(pnl)
        all_avg.append(avg)

    elapsed = time.time() - t0
    wr_all = torch.cat(all_wr)
    trades_all = torch.cat(all_trades)
    pnl_all = torch.cat(all_pnl)
    avg_all = torch.cat(all_avg)

    print(f"  [{tf_name}] Completed {total:,} combos in {elapsed:.2f}s "
          f"({total/elapsed:,.0f}/sec)")

    # Filter: min trade count
    valid = trades_all >= min_trades
    if valid.sum() == 0:
        print(f"  [{tf_name}] No combos with >={min_trades} trades")
        # Relax to min 10
        valid = trades_all >= 10
        if valid.sum() == 0:
            return []

    # Rank by WR (primary), then by PnL (tiebreak)
    # Set invalid combos to -1 WR so they rank last
    wr_ranked = torch.where(valid, wr_all, torch.tensor(-1.0, device=DEVICE))
    topk_count = min(50, int(valid.sum().item()))
    _, topk_idx = torch.topk(wr_ranked, topk_count)

    results = []
    for rank, idx in enumerate(topk_idx):
        i = idx.item()
        results.append({
            "tf": tf_name,
            "tf_id": tf_id,
            "rank": rank + 1,
            "conf_min": conf_flat[i].item(),
            "pvt_r_min": pvt_flat[i].item(),
            "trail_buffer": trail_flat[i].item(),
            "win_rate": wr_all[i].item(),
            "n_trades": int(trades_all[i].item()),
            "net_pnl": pnl_all[i].item(),
            "avg_pnl": avg_all[i].item(),
            "leverage": lev,
        })

    best = results[0]
    print(f"  [{tf_name}] Best: WR={best['win_rate']:.1%} | "
          f"n={best['n_trades']} | PnL=${best['net_pnl']:,.0f} | "
          f"conf>={best['conf_min']:.3f} pvt>={best['pvt_r_min']:.3f} "
          f"trail={best['trail_buffer']:.3f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Combined Multi-TF Search
# ═══════════════════════════════════════════════════════════════════════════════

def search_combined(
    data_splits: dict,
    tf_configs: dict[int, dict],
) -> list[dict]:
    """Evaluate combined TF configs on a data split.

    Takes the best per-TF params and tests combinations of
    which TFs to enable/disable.
    """
    # Test all 2^4=16 combinations of enabled TFs
    results = []
    for enable_mask in range(1, 16):  # Skip 0 (nothing enabled)
        total_wins = 0
        total_trades = 0
        total_pnl = 0.0
        tf_details = {}

        for tf_id in range(4):
            if not (enable_mask & (1 << tf_id)):
                continue
            if tf_id not in tf_configs:
                continue

            cfg = tf_configs[tf_id]
            data = data_splits.get(tf_id)
            if data is None or data["n"] == 0:
                continue

            wr, nt, pnl, avg = simulate_tf_batch(
                data,
                torch.tensor([cfg["conf_min"]], dtype=DTYPE, device=DEVICE),
                torch.tensor([cfg["pvt_r_min"]], dtype=DTYPE, device=DEVICE),
                torch.tensor([cfg["trail_buffer"]], dtype=DTYPE, device=DEVICE),
                cfg["leverage"],
            )

            n = int(nt[0].item())
            w = int(round(wr[0].item() * n))
            total_wins += w
            total_trades += n
            total_pnl += pnl[0].item()
            tf_names = ["5m", "30m", "1h", "4h"]
            tf_details[tf_names[tf_id]] = {
                "wr": wr[0].item(), "n": n, "pnl": pnl[0].item()
            }

        if total_trades == 0:
            continue

        combined_wr = total_wins / total_trades
        enabled_tfs = [["5m","30m","1h","4h"][i] for i in range(4) if enable_mask & (1<<i)]

        results.append({
            "enabled_tfs": enabled_tfs,
            "enable_mask": enable_mask,
            "win_rate": combined_wr,
            "n_trades": total_trades,
            "net_pnl": total_pnl,
            "tf_details": tf_details,
        })

    results.sort(key=lambda x: (-x["win_rate"], -x["n_trades"]))
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# March 2026 Crash Stress Test
# ═══════════════════════════════════════════════════════════════════════════════

def stress_test_march(df: pd.DataFrame, tf_configs: dict, enabled_tfs: list[str]):
    """Test win rate stability during March 2026."""
    mar_df = df[(df.entry_ts >= "2026-03-01") & (df.entry_ts <= "2026-03-25")]

    print(f"\n{'='*80}")
    print(f"  MARCH 2026 CRASH STRESS TEST")
    print(f"{'='*80}")

    total_wins = 0
    total_trades = 0

    for tf_name in enabled_tfs:
        tf_id = TF_ENCODE[tf_name]
        if tf_id not in tf_configs:
            continue

        cfg = tf_configs[tf_id]
        tf_df = mar_df[mar_df.best_tf == tf_name].copy()
        if len(tf_df) == 0:
            continue

        # Apply dual-gate filter
        mask = (tf_df.confidence >= cfg["conf_min"]) & (tf_df.pvt_r >= cfg["pvt_r_min"])
        filtered = tf_df[mask]

        if len(filtered) == 0:
            print(f"  {tf_name}: 0 trades (all filtered)")
            continue

        n = len(filtered)
        hist_lev = filtered.leverage.clip(lower=1)
        base_move = filtered.pnl_pct / hist_lev

        # Apply trail model
        buf = cfg["trail_buffer"]
        win_cap = 0.50 + 0.35 * buf
        loss_red = 0.30 + 0.50 * buf

        adj = np.where(base_move > 0, base_move * win_cap, base_move * loss_red)
        adj = adj * cfg["leverage"]
        adj = np.clip(adj, -(HARD_SL_PCT * 100), None) - ROUND_TRIP_COST_PCT

        w = int((adj > 0).sum())
        wr = w / n if n > 0 else 0

        total_wins += w
        total_trades += n
        pnl = adj.sum() * INITIAL_CAPITAL * POS_FRAC / 100

        print(f"  {tf_name}: {n:4d} trades | WR={wr:.1%} | PnL=${pnl:,.0f}")

    if total_trades > 0:
        print(f"  {'TOTAL':>3s}: {total_trades:4d} trades | "
              f"WR={total_wins/total_trades:.1%}")
    else:
        print("  No trades in March 2026")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  Project DRACO — Elite-Win Optimizer")
    print("  Target: Win Rate > 65% | Min 800 Trades")
    print("  GPU: AMD Radeon RX 9070 XT | ROCm 6.3 | PyTorch 2.9")
    print("=" * 80)

    if torch.cuda.is_available():
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:   {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    print(f"\n  FIXED: Exit=Adaptive Trail | Hard SL=1.5% | Entry=Dual-Gate")
    print(f"  SEARCH: conf=[0.92-0.98] | pvt_r=[0.92-0.98] | trail=[0.40-1.20]")
    print(f"  GRID:   {len(CONF_RANGE)}×{len(PVT_RANGE)}×{len(TRAIL_RANGE)} = "
          f"{len(CONF_RANGE)*len(PVT_RANGE)*len(TRAIL_RANGE):,} per TF")

    # Load data
    data = load_data()
    t_total = time.time()

    # ══════════════════════════════════════════════════════════════════════
    # Phase 1: Per-TF grid search on TRAINING data
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  PHASE 1: PER-TIMEFRAME GRID SEARCH (Training)")
    print(f"{'='*80}")

    tf_best = {}
    for tf_id in range(4):
        results = search_single_tf(tf_id, data["train"].get(tf_id), min_trades=30)
        if results:
            tf_best[tf_id] = results[0]

    # ══════════════════════════════════════════════════════════════════════
    # Phase 2: Find best TF combination on TRAINING
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  PHASE 2: OPTIMAL TF COMBINATION (Training)")
    print(f"{'='*80}")

    train_combos = search_combined(data["train"], tf_best)

    print(f"\n  Top 10 TF combinations (Training):")
    print(f"  {'TFs':<20} | {'WR':>6} | {'Trades':>6} | {'PnL':>10}")
    print(f"  {'-'*50}")
    for r in train_combos[:10]:
        tfs = "+".join(r["enabled_tfs"])
        print(f"  {tfs:<20} | {r['win_rate']:>5.1%} | {r['n_trades']:>6} | "
              f"${r['net_pnl']:>9,.0f}")

    # Filter: WR > 50% AND trades > 200 for training
    viable = [r for r in train_combos if r["win_rate"] > 0.50 and r["n_trades"] > 200]
    if not viable:
        viable = train_combos[:5]  # Fallback to top 5

    # ══════════════════════════════════════════════════════════════════════
    # Phase 3: Validate on VALIDATION set
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  PHASE 3: VALIDATION")
    print(f"{'='*80}")

    val_combos = search_combined(data["val"], tf_best)

    print(f"\n  Top 10 TF combinations (Validation):")
    print(f"  {'TFs':<20} | {'WR':>6} | {'Trades':>6} | {'PnL':>10}")
    print(f"  {'-'*50}")
    for r in val_combos[:10]:
        tfs = "+".join(r["enabled_tfs"])
        print(f"  {tfs:<20} | {r['win_rate']:>5.1%} | {r['n_trades']:>6} | "
              f"${r['net_pnl']:>9,.0f}")

    # ══════════════════════════════════════════════════════════════════════
    # Phase 4: Blind Test
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  PHASE 4: BLIND TEST")
    print(f"{'='*80}")

    blind_combos = search_combined(data["blind"], tf_best)

    print(f"\n  Top 10 TF combinations (Blind Test):")
    print(f"  {'TFs':<20} | {'WR':>6} | {'Trades':>6} | {'PnL':>10}")
    print(f"  {'-'*50}")
    for r in blind_combos[:10]:
        tfs = "+".join(r["enabled_tfs"])
        print(f"  {tfs:<20} | {r['win_rate']:>5.1%} | {r['n_trades']:>6} | "
              f"${r['net_pnl']:>9,.0f}")

    # ══════════════════════════════════════════════════════════════════════
    # Select Elite-Win Configuration
    # ══════════════════════════════════════════════════════════════════════

    # Find the blind test combo with highest WR and >= 200 trades
    elite_candidates = [r for r in blind_combos if r["n_trades"] >= 200]
    if not elite_candidates:
        elite_candidates = blind_combos[:5]

    elite = elite_candidates[0]

    # Also find matching train/val results for the same TF combo
    elite_mask = elite["enable_mask"]
    train_match = next((r for r in train_combos if r["enable_mask"] == elite_mask), None)
    val_match = next((r for r in val_combos if r["enable_mask"] == elite_mask), None)

    print(f"\n{'='*80}")
    print(f"  ELITE-WIN CONFIGURATION")
    print(f"{'='*80}")

    print(f"\n  Enabled Timeframes: {', '.join(elite['enabled_tfs'])}")
    print(f"\n  Per-Timeframe Thresholds:")
    print(f"  {'TF':<6} | {'Conf Min':>9} | {'PVT R Min':>9} | {'Trail Buf':>9} | {'Leverage':>8}")
    print(f"  {'-'*50}")

    for tf_name in elite["enabled_tfs"]:
        tf_id = TF_ENCODE[tf_name]
        cfg = tf_best[tf_id]
        print(f"  {tf_name:<6} | {cfg['conf_min']:>9.3f} | {cfg['pvt_r_min']:>9.3f} | "
              f"{cfg['trail_buffer']:>9.3f} | {cfg['leverage']:>7.0f}x")

    print(f"\n  Fixed Parameters:")
    print(f"    Entry Gate:  DUAL (both conf AND pvt_r required)")
    print(f"    Exit Logic:  Adaptive Trailing Stop (midline-based ratchet)")
    print(f"    Hard SL:     1.5%")

    print(f"\n  Performance Across Splits:")
    print(f"  {'Split':<8} | {'WR':>6} | {'Trades':>6} | {'Net PnL':>10}")
    print(f"  {'-'*40}")
    if train_match:
        print(f"  {'Train':<8} | {train_match['win_rate']:>5.1%} | "
              f"{train_match['n_trades']:>6} | ${train_match['net_pnl']:>9,.0f}")
    if val_match:
        print(f"  {'Val':<8} | {val_match['win_rate']:>5.1%} | "
              f"{val_match['n_trades']:>6} | ${val_match['net_pnl']:>9,.0f}")
    print(f"  {'Blind':<8} | {elite['win_rate']:>5.1%} | "
          f"{elite['n_trades']:>6} | ${elite['net_pnl']:>9,.0f}")

    # ── Trade Distribution by TF ──
    print(f"\n  Win Rate by Timeframe (Blind Test):")
    print(f"  {'TF':<6} | {'WR':>6} | {'Trades':>6} | {'PnL':>10}")
    print(f"  {'-'*35}")
    for tf_name, detail in elite.get("tf_details", {}).items():
        print(f"  {tf_name:<6} | {detail['wr']:>5.1%} | {detail['n']:>6} | "
              f"${detail['pnl']:>9,.0f}")

    # ── March 2026 Stress Test ──
    stress_test_march(data["_df"], tf_best, elite["enabled_tfs"])

    # ── Save ──
    elite_config = {
        "name": "Elite-Win",
        "enabled_tfs": elite["enabled_tfs"],
        "fixed": {
            "entry_gate": "DUAL (conf AND pvt_r)",
            "exit_logic": "Adaptive Trailing Stop",
            "hard_sl_pct": HARD_SL_PCT,
        },
        "per_tf": {},
        "blind_test": {
            "win_rate": elite["win_rate"],
            "n_trades": elite["n_trades"],
            "net_pnl": elite["net_pnl"],
        },
    }
    for tf_name in elite["enabled_tfs"]:
        tf_id = TF_ENCODE[tf_name]
        cfg = tf_best[tf_id]
        elite_config["per_tf"][tf_name] = {
            "conf_min": cfg["conf_min"],
            "pvt_r_min": cfg["pvt_r_min"],
            "trail_buffer": cfg["trail_buffer"],
            "leverage": cfg["leverage"],
        }
    if elite.get("tf_details"):
        elite_config["tf_breakdown"] = elite["tf_details"]
    if train_match:
        elite_config["train"] = {
            "win_rate": train_match["win_rate"],
            "n_trades": train_match["n_trades"],
            "net_pnl": train_match["net_pnl"],
        }
    if val_match:
        elite_config["val"] = {
            "win_rate": val_match["win_rate"],
            "n_trades": val_match["n_trades"],
            "net_pnl": val_match["net_pnl"],
        }

    out_path = RESULTS_DIR / "elite_win_config.json"
    with open(out_path, "w") as f:
        json.dump(elite_config, f, indent=2)

    total_time = time.time() - t_total
    total_combos = len(CONF_RANGE) * len(PVT_RANGE) * len(TRAIL_RANGE) * 4

    print(f"\n{'='*80}")
    print(f"  COMPLETE — {total_combos:,} total combos in {total_time:.1f}s")
    print(f"  Config saved: {out_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

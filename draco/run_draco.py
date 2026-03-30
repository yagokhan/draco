#!/usr/bin/env python3
"""
Project DRACO — Main Entry Point.

GPU-accelerated hybrid trading optimizer for AMD Radeon RX 9070 XT.

Usage:
    cd /home/yagokhan/draco
    source /home/yagokhan/chameleon/claude_code_project/algo_env/bin/activate
    python3 run_draco.py

Environment:
    ROCR_VISIBLE_DEVICES=0  (force 9070 XT as primary GPU)
"""
from __future__ import annotations
import os
import sys
import time
import torch

# Ensure we're in the right directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DEVICE, GOLD_CSV, RAW_CSV
from signal_auditor import load_trade_universe
from optimizer import run_optimization


def main():
    print("=" * 80)
    print("  Project DRACO — GPU-Accelerated Hybrid Trading Optimizer")
    print("  AMD Radeon RX 9070 XT | ROCm 6.3 | PyTorch 2.9")
    print("=" * 80)

    # ── Hardware check ──
    print(f"\n[Hardware]")
    print(f"  PyTorch:  {torch.__version__}")
    print(f"  ROCm:     {torch.version.hip}")
    print(f"  Device:   {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU:      {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory
        print(f"  VRAM:     {mem / 1024**3:.1f} GB")
    else:
        print("  WARNING: No GPU detected — running on CPU (will be slow)")

    # ── Data check ──
    print(f"\n[Data Sources]")
    print(f"  Gold Standard: {GOLD_CSV}")
    print(f"  Raw Extended:  {RAW_CSV}")

    if not GOLD_CSV.exists():
        print(f"  ERROR: Gold Standard CSV not found at {GOLD_CSV}")
        sys.exit(1)
    if not RAW_CSV.exists():
        print(f"  ERROR: Raw Extended CSV not found at {RAW_CSV}")
        sys.exit(1)

    # ── Load data ──
    print(f"\n[Loading Trade Universe]")
    t0 = time.time()
    data_splits = load_trade_universe()
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ── GPU warmup ──
    if torch.cuda.is_available():
        print(f"\n[GPU Warmup]")
        x = torch.randn(1000, 1000, device=DEVICE)
        _ = torch.mm(x, x)
        torch.cuda.synchronize()
        print(f"  GPU ready")

    # ── Run optimization ──
    t_total = time.time()
    results = run_optimization(data_splits)
    total_time = time.time() - t_total

    # ── Summary ──
    print("\n" + "=" * 80)
    print(f"  DRACO COMPLETE — Total time: {total_time:.1f}s")
    print("=" * 80)

    if results and "alpha" in results:
        alpha = results["alpha"]
        print(f"\n  #1 Configuration: Aeternus-Alpha")
        print(f"  Net PnL:    ${alpha['net_pnl']:,.2f}")
        print(f"  Win Rate:   {alpha['win_rate']:.1%}")
        print(f"  Max DD:     {alpha['max_dd']:.1%}")
        print(f"  Calmar:     {alpha['calmar']:.2f}")
        print(f"\n  Results: ~/draco/results/draco_results.json")
        print(f"  Alpha:   ~/draco/results/draco_alpha.json")


if __name__ == "__main__":
    main()

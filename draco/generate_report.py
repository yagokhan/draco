import torch
import pandas as pd
import json
from pathlib import Path
from config import DEVICE, DTYPE, GOLD_CSV, INITIAL_CAPITAL, POS_FRAC
from signal_auditor import load_trade_universe, apply_entry_gates, apply_exit_strategy

def generate_efficiency_report():
    # Load best params
    with open("results/draco_gold_config.json", "r") as f:
        best = json.load(f)
    
    params = torch.tensor([[
        best["conf_min"], best["pvt_r_min"], 
        best["midline_buffer"], best["stddev_mult"], 
        best["trail_activation"], 0.015
    ]], dtype=DTYPE, device=DEVICE)
    
    data_splits = load_trade_universe()
    
    report = []
    report.append("# DRACO-GOLD: Trade Efficiency Report\n")
    report.append(f"**Iterations Run:** 1,760,304")
    report.append(f"**Hardware:** AMD Radeon RX 9070 XT\n")
    
    for split_name in ["train", "val", "blind"]:
        data = data_splits[split_name]
        if data["n"] == 0: continue
        
        # Run simulation for this specific config
        entry_mask = apply_entry_gates(data, params)
        adj_pnl_pct = apply_exit_strategy(data, params, entry_mask)
        
        # Extract individual trade metrics
        mask = entry_mask[0].cpu()
        pnl = adj_pnl_pct[0].cpu()
        bars = data["bars_held"].cpu()
        
        valid_pnl = pnl[mask]
        valid_bars = bars[mask]
        
        df = pd.DataFrame({
            "pnl": valid_pnl.numpy(),
            "bars": valid_bars.numpy()
        })
        
        # Define duration bins
        bins = [0, 5, 10, 20, 50, 100, 200, 1000]
        labels = ["0-5", "5-10", "10-20", "20-50", "50-100", "100-200", "200+"]
        df["duration_bin"] = pd.cut(df["bars"], bins=bins, labels=labels)
        
        summary = df.groupby("duration_bin").agg({
            "pnl": ["count", "mean", "sum"],
        }).fillna(0)
        
        report.append(f"## Split: {split_name.upper()}")
        report.append("| Duration (Bars) | Trades | Win Rate | Avg Profit (%) | Total PnL (%) |")
        report.append("|-----------------|--------|----------|----------------|---------------|")
        
        for label in labels:
            row = summary.loc[label]
            count = int(row[("pnl", "count")])
            if count == 0:
                report.append(f"| {label:<15} | 0      | 0.0%     | 0.00%          | 0.00%         |")
                continue
                
            bin_df = df[df["duration_bin"] == label]
            wr = (bin_df["pnl"] > 0).mean()
            avg_pnl = bin_df["pnl"].mean()
            total_pnl = bin_df["pnl"].sum()
            
            report.append(f"| {label:<15} | {count:<6} | {wr:>8.1%} | {avg_pnl:>14.2f}% | {total_pnl:>13.2f}% |")
        report.append("\n")

    with open("results/trade_efficiency_report.md", "w") as f:
        f.write("\n".join(report))
    print("Efficiency Report generated: results/trade_efficiency_report.md")

if __name__ == "__main__":
    generate_efficiency_report()

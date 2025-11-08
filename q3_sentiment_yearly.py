import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def parse_args():
    ap = argparse.ArgumentParser(description="Q3 yearly sentiment vs nutrients (ENERC_kcal) - light version")
    ap.add_argument("--sent-root", default="/projappl/project_2015109/work/results_q2",
                    help="Root folder where Q2 results are saved (contains subfolders sentiment_mean/, etc.)")
    ap.add_argument("--q1-base", default="/projappl/project_2015109/work/Q1_outputs",
                    help="Folder containing monthly_nutrients.parquet (from Q1).")
    ap.add_argument("--sent-stat", default="sentiment_mean",
                    choices=["sentiment_mean", "sentiment_sum", "sentiment_median"],
                    help="Which sentiment statistic to use at yearly level.")
    ap.add_argument("--out-root", default="/projappl/project_2015109/work",
                    help="Base output root (Q3 results will be under {out-root}/results_q3/<sent-stat>/)")
    return ap.parse_args()

def main():
    args = parse_args()

    monthly_sent_p = os.path.join(args.sent_root, args.sent_stat, "monthly_sentiment.parquet")
    monthly_nutr_p = os.path.join(args.q1_base, "monthly_nutrients.parquet")
    out_dir = os.path.join(args.out_root, "results_q3", args.sent_stat)
    ensure_dir(out_dir)

    log(f"[read] {monthly_sent_p}")
    if not os.path.exists(monthly_sent_p):
        log(f"[error] Monthly sentiment file not found: {monthly_sent_p}")
        sys.exit(1)

    log(f"[read] {monthly_nutr_p}")
    if not os.path.exists(monthly_nutr_p):
        log(f"[error] Monthly nutrients file not found: {monthly_nutr_p}")
        sys.exit(1)

    sent = pd.read_parquet(monthly_sent_p)
    nutr = pd.read_parquet(monthly_nutr_p)

    if "month" not in sent.columns or args.sent_stat not in sent.columns:
        log(f"[error] Sentiment table must contain 'month' and '{args.sent_stat}'. Columns found: {list(sent.columns)[:8]} ...")
        sys.exit(1)

    if "month" not in nutr.columns or "ENERC_kcal" not in nutr.columns:
        log(f"[error] Nutrient table must contain 'month' and 'ENERC_kcal'. Columns found: {list(nutr.columns)[:12]} ...")
        sys.exit(1)

    sent = sent[["month", args.sent_stat]].copy()
    nutr = nutr[["month", "ENERC_kcal"]].copy()
    sent["year"] = sent["month"].astype(str).str[:4]
    nutr["year"] = nutr["month"].astype(str).str[:4]

    log(f"[aggregate] Sentiment yearly using: {args.sent_stat}")
    if args.sent_stat == "sentiment_sum":
        yearly_sent = sent.groupby("year", as_index=False)[args.sent_stat].sum()
    else:
        yearly_sent = sent.groupby("year", as_index=False)[args.sent_stat].mean()

    log("[aggregate] Nutrients yearly using sum of months")
    yearly_nutr = nutr.groupby("year", as_index=False)["ENERC_kcal"].sum()

    yearly = pd.merge(yearly_sent, yearly_nutr, on="year", how="inner")
    yearly = yearly.sort_values("year").reset_index(drop=True)

    log(f"[years] years={len(yearly)} range={yearly['year'].min()}–{yearly['year'].max()}")

    a = pd.to_numeric(yearly[args.sent_stat], errors="coerce")
    b = pd.to_numeric(yearly["ENERC_kcal"], errors="coerce")
    m = a.notna() & b.notna()
    if m.sum() < 2:
        r = float("nan")
    else:
        r = float(np.corrcoef(a[m], b[m])[0, 1])

    log(f"[corr] Pearson r between {args.sent_stat} and ENERC_kcal (yearly): {r:.4f}")

    y_sent_p = os.path.join(out_dir, "yearly_sentiment.parquet")
    y_merged_p = os.path.join(out_dir, "yearly_sentiment_vs_nutrients.parquet")
    summary_p = os.path.join(out_dir, "q3_summary.txt")
    plot_p = os.path.join(out_dir, "sentiment_vs_ENERC_kcal_yearly.png")

    yearly_sent.to_parquet(y_sent_p, index=False)
    log(f"[write] {y_sent_p}")

    yearly.to_parquet(y_merged_p, index=False)
    log(f"[write] {y_merged_p}")

    with open(summary_p, "w", encoding="utf-8") as f:
        f.write(f"Sentiment column (yearly): {args.sent_stat}\n")
        f.write("Nutrient: ENERC_kcal (yearly sum)\n")
        f.write(f"Pearson r (yearly): {r:.6f}\n")
        f.write(f"Years: {yearly['year'].min()}–{yearly['year'].max()} (n={len(yearly)})\n")
    log(f"[write] {summary_p}")

    log("[plot] drawing yearly sentiment (left axis) vs ENERC_kcal (right axis)")
    ax1 = plt.subplots(figsize=(10, 5))

    years = yearly["year"].astype(str).tolist()
    x = np.arange(len(years))

    sienna = "#A0522D"
    teal = "#008080"

    ax1.plot(x, yearly[args.sent_stat].values, color=sienna, marker="o", linewidth=2, label=args.sent_stat)
    ax1.set_xlabel("Year")
    ax1.set_ylabel(args.sent_stat, color=sienna)
    ax1.tick_params(axis="y", labelcolor=sienna)
    ax1.set_xticks(x)
    ax1.set_xticklabels(years, rotation=45, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(x, yearly["ENERC_kcal"].values, color=teal, marker="s", linestyle="--", linewidth=2, label="ENERC_kcal")
    ax2.set_ylabel("ENERC_kcal", color=teal)
    ax2.tick_params(axis="y", labelcolor=teal)

    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="upper left")

    ax1.grid(True, linestyle=":", alpha=0.5)
    plt.title(f"Yearly Sentiment score vs nutrition intake (r = {r:.3f})")
    plt.tight_layout()
    plt.savefig(plot_p, dpi=200)
    plt.close()
    log(f"[write] {plot_p}")

    log("[done] Q3 yearly pipeline complete.")

if __name__ == "__main__":
    main()

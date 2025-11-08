import argparse
import collections
import glob
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(stage, msg):
    print(f"[{ts()}] [{stage}] {msg}", flush=True)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

# def month_key_sortable(mstr):
#     # 'YYYY-MM' sorts lexicographically already, but keep helper for clarity
#     return mstr

def compute_pearson(a, b):
    a = pd.Series(a, dtype="float64")
    b = pd.Series(b, dtype="float64")
    mask = a.notna() & b.notna()
    if mask.sum() < 2:
        return np.nan
    return float(np.corrcoef(a[mask], b[mask])[0, 1])

def plot_dual_axis_monthly(df, nutrient, out_png):
    """
    df: columns ['month', 'foods_distinct', <nutrient>]
    Wider figure + quarterly ticks for readability.
    """
    months = df["month"].astype(str).tolist()
    x_idx = np.arange(len(months))
    y_div = df["foods_distinct"].astype(float).values
    y_nut = df[nutrient].astype(float).values

    quarter_tick_idx = [i for i, m in enumerate(months) if m.endswith(("-01", "-04", "-07", "-10"))]
    quarter_tick_lbl = [months[i] for i in quarter_tick_idx]

    fig, ax1 = plt.subplots(figsize=(20, 8))
    color_div = 'tab:blue'
    color_nut = 'tab:orange'

    l1, = ax1.plot(x_idx, y_div, label="Distinct foods (monthly)", color=color_div, linewidth=1.8)
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Distinct foods (monthly)", color=color_div)
    ax1.tick_params(axis='y', labelcolor=color_div)
    ax1.grid(True, axis='y', alpha=0.25)

    ax1.set_xticks(quarter_tick_idx)
    ax1.set_xticklabels(quarter_tick_lbl, rotation=45, ha='right', fontsize=8)

    ax2 = ax1.twinx()
    l2, = ax2.plot(x_idx, y_nut, label=f"{nutrient} (monthly)", color=color_nut, linewidth=1.8)
    ax2.set_ylabel(f"{nutrient} (monthly)", color=color_nut)
    ax2.tick_params(axis='y', labelcolor=color_nut)
    ax2.grid(False)

    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")

    plt.title("Monthly food diversity and nutrition intake")
    fig.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)



def plot_dual_axis_yearly(df, nutrient, out_png):
    """
    df: columns ['year', 'foods_distinct_year', <nutrient>]
    """
    x = df["year"].tolist()
    y_div = df["foods_distinct_year"].astype(float).values
    y_nut = df[nutrient].astype(float).values

    fig, ax1 = plt.subplots(figsize=(12, 5.5))
    color_div = 'tab:blue'
    color_nut = 'tab:orange'

    l1, = ax1.plot(x, y_div, label="Distinct foods (yearly)", color=color_div, marker='o', linewidth=1.8)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Distinct foods (yearly)", color=color_div)
    ax1.tick_params(axis='y', labelcolor=color_div)
    ax1.grid(True, axis='y', alpha=0.25)

    ax2 = ax1.twinx()
    l2, = ax2.plot(x, y_nut, label=f"{nutrient} (yearly sum)", color=color_nut, marker='o', linewidth=1.8)
    ax2.set_ylabel(f"{nutrient} (yearly sum)", color=color_nut)
    ax2.tick_params(axis='y', labelcolor=color_nut)

    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")

    plt.title("Yearly food diversity and nutrition intake")
    fig.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

def scan_threads_for_diversity(threads_glob):
    """
    Returns:
      monthly_df: columns ['month','foods_distinct','foods_mentions','lines']
      yearly_df:  columns ['year','foods_distinct_year','foods_mentions_year','lines_year']
    """
    by_month = collections.defaultdict(lambda: {"distinct": set(), "mentions": 0, "lines": 0})
    by_year  = collections.defaultdict(lambda: {"distinct": set(), "mentions": 0, "lines": 0})

    files = sorted(glob.glob(threads_glob))
    if not files:
        raise FileNotFoundError(f"No files matched: {threads_glob}")

    total_lines = 0
    for p in files:
        log("read", p)
        with open(p, encoding="utf-8") as f:
            for line in f:
                total_lines += 1
                r = json.loads(line)
                m = r.get("month")
                if not m:
                    continue
                y = str(m)[:4]
                ids = r.get("food_ids") or []
                by_month[m]["distinct"].update(ids)
                by_month[m]["mentions"] += len(ids)
                by_month[m]["lines"]    += 1
                by_year[y]["distinct"].update(ids)
                by_year[y]["mentions"] += len(ids)
                by_year[y]["lines"]    += 1

    log("scan", f"lines processed: {total_lines:,}")

    rows_m = []
    # for m in sorted(by_month.keys(), key=month_key_sortable):
    for m in sorted(by_month.keys()):
        d = by_month[m]
        rows_m.append({
            "month": m,
            "foods_distinct": len(d["distinct"]),
            "foods_mentions": d["mentions"],
            "lines": d["lines"],
        })
    monthly_df = pd.DataFrame(rows_m)

    rows_y = []
    for y in sorted(by_year.keys()):
        d = by_year[y]
        rows_y.append({
            "year": y,
            "foods_distinct_year": len(d["distinct"]),
            "foods_mentions_year": d["mentions"],
            "lines_year": d["lines"],
        })
    yearly_df = pd.DataFrame(rows_y)

    return monthly_df, yearly_df


def run_monthly(base, nutrient, outdir):
    threads_glob = os.path.join(base, "processed", "s24_year_threads", "threads_20*.jsonl")
    q1_nutr_path = os.path.join(base, "Q1_outputs", "monthly_nutrients.parquet")

    monthly_df, _ = scan_threads_for_diversity(threads_glob)

    log("monthly", f"rows={len(monthly_df)} months={monthly_df['month'].nunique()}")
    out_m_div = os.path.join(outdir, "monthly_food_diversity.parquet")
    monthly_df.to_parquet(out_m_div, index=False)
    log("write", f"{out_m_div}")

    log("read", q1_nutr_path)
    nutr = pd.read_parquet(q1_nutr_path)
    if nutrient not in nutr.columns:
        raise ValueError(f"Nutrient column '{nutrient}' not found in {q1_nutr_path}")

    merged = pd.merge(monthly_df, nutr[["month", nutrient]], on="month", how="inner")
    merged = merged.sort_values("month").reset_index(drop=True)

    out_m_merged = os.path.join(outdir, "monthly_diversity_vs_nutrients.parquet")
    merged.to_parquet(out_m_merged, index=False)
    log("write", f"{out_m_merged}")

    r = compute_pearson(merged["foods_distinct"], merged[nutrient])
    log("corr", f"Pearson r (monthly) between foods_distinct and {nutrient}: {r:.4f}")

    out_png = os.path.join(outdir, f"monthly_diversity_vs_{nutrient}.png")
    plot_dual_axis_monthly(merged[["month", "foods_distinct", nutrient]], nutrient, out_png)
    log("write", out_png)

    return r, out_png


def run_yearly(base, nutrient, outdir):
    threads_glob = os.path.join(base, "processed", "s24_year_threads", "threads_20*.jsonl")
    q1_nutr_path = os.path.join(base, "Q1_outputs", "monthly_nutrients.parquet")

    _, yearly_df = scan_threads_for_diversity(threads_glob)

    log("yearly", f"rows={len(yearly_df)} years={yearly_df['year'].nunique()}")
    out_y_div = os.path.join(outdir, "yearly_food_diversity.parquet")
    yearly_df.to_parquet(out_y_div, index=False)
    log("write", f"{out_y_div}")

    log("read", q1_nutr_path)
    nutr_m = pd.read_parquet(q1_nutr_path)
    if nutrient not in nutr_m.columns:
        raise ValueError(f"Nutrient column '{nutrient}' not found in {q1_nutr_path}")
    nutr_m = nutr_m[["month", nutrient]].copy()
    nutr_m["year"] = nutr_m["month"].str[:4]
    nutr_y = nutr_m.groupby("year", as_index=False)[nutrient].sum()

    merged = pd.merge(yearly_df, nutr_y, on="year", how="inner")
    merged = merged.sort_values("year").reset_index(drop=True)

    out_y_merged = os.path.join(outdir, "yearly_diversity_vs_nutrients.parquet")
    merged.to_parquet(out_y_merged, index=False)
    log("write", f"{out_y_merged}")

    r = compute_pearson(merged["foods_distinct_year"], merged[nutrient])
    log("corr", f"Pearson r (yearly) between foods_distinct_year and {nutrient}: {r:.4f}")

    out_png = os.path.join(outdir, f"yearly_diversity_vs_{nutrient}.png")
    plot_dual_axis_yearly(merged[["year", "foods_distinct_year", nutrient]], nutrient, out_png)
    log("write", out_png)

    return r, out_png

def main():
    ap = argparse.ArgumentParser(description="Q4: Food diversity vs nutrients (monthly & yearly)")
    ap.add_argument("--base", required=True,
                    help="Base work dir (expects processed/s24_year_threads and Q1_outputs/monthly_nutrients.parquet under it)")
    ap.add_argument("--nutrient", default="ENERC_kcal",
                    help="Nutrient column to compare against (default: ENERC_kcal)")
    ap.add_argument("--outdir-name", default="results_q4",
                    help="Folder name under --base for outputs (default: results_q4)")
    ap.add_argument("--monthly", action="store_true", help="Run monthly analysis only")
    ap.add_argument("--yearly", action="store_true", help="Run yearly analysis only")
    args = ap.parse_args()

    base = os.path.abspath(args.base)
    outdir = ensure_dir(os.path.join(base, args.outdir_name))

    log("start", f"base={base}")
    log("param", f"nutrient={args.nutrient}")

    run_m = args.monthly or not (args.monthly or args.yearly)
    run_y = args.yearly or not (args.monthly or args.yearly)

    summary_lines = []
    summary_path = os.path.join(outdir, "q4_summary.txt")

    if run_m:
        r_m, png_m = run_monthly(base, args.nutrient, outdir)
        summary_lines.append(f"Nutrient: {args.nutrient} | Pearson r (monthly): {r_m:.6f}")
        summary_lines.append(f"Plot (monthly): {png_m}")

    if run_y:
        r_y, png_y = run_yearly(base, args.nutrient, outdir)
        summary_lines.append(f"Nutrient: {args.nutrient} | Pearson r (yearly): {r_y:.6f}")
        summary_lines.append(f"Plot (yearly): {png_y}")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    log("write", summary_path)
    log("done", "Q4 diversity pipeline complete.")

if __name__ == "__main__":
    main()

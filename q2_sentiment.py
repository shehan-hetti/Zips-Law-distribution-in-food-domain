import argparse
import os
import sys
import glob
from datetime import datetime

import pandas as pd
import numpy as np
from afinn import Afinn
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def log(msg):
    """
    Simple log to stdout
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def find_one(patterns):
    for pat in patterns:
        hits = glob.glob(pat, recursive=True)
        if hits:
            hits.sort(key=lambda p: os.path.getsize(p), reverse=True)
            return hits[0]
    return None


def load_monthly_titles(base_dir):
    """
    Load monthly_titles.parquet (month, thread_id, title).
    """
    candidate = find_one([
        os.path.join(base_dir, "**", "monthly_titles.parquet"),
        os.path.join(base_dir, "monthly_titles.parquet"),
    ])
    if not candidate or not os.path.exists(candidate):
        raise FileNotFoundError(
            f"monthly_titles.parquet not found under {base_dir}.\n"
            f"Please provide the path via --titles."
        )
    log(f"[read] {candidate}")
    df = pd.read_parquet(candidate)
    want_cols = {"month", "thread_id", "title"}
    missing = want_cols - set(df.columns)
    if missing:
        raise ValueError(f"monthly_titles.parquet missing columns: {missing}")
    df = df.copy()
    df["month"] = df["month"].astype(str)
    df["thread_id"] = df["thread_id"].astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    log(f"[titles] rows={len(df):,} months={df['month'].nunique()}")
    return df


def load_monthly_nutrients(base_dir):
    """
    Load monthly_nutrients.parquet (month, food_count, nutrients...).
    """
    candidate = find_one([
        os.path.join(base_dir, "**", "monthly_nutrients.parquet"),
        os.path.join(base_dir, "monthly_nutrients.parquet"),
    ])
    if not candidate or not os.path.exists(candidate):
        raise FileNotFoundError(
            f"monthly_nutrients.parquet not found under {base_dir}.\n"
            f"Please provide the path via --nutrients."
        )
    log(f"[read] {candidate}")
    df = pd.read_parquet(candidate)
    if "month" not in df.columns:
        raise ValueError("monthly_nutrients.parquet missing 'month' column")
    df = df.copy()
    df["month"] = df["month"].astype(str)
    log(f"[nutrients] rows={len(df):,} months={df['month'].nunique()} cols={len(df.columns)}")
    return df


def score_titles_monthly(titles_df, afinn_lang="fi", emoticons=False):
    """
    Compute AFINN sentiment for each title, aggregate per month.
    Return dataframe with monthly aggregates.
    """
    log(f"[afinn] loading lexicon lang='{afinn_lang}' emoticons={emoticons}")
    af = Afinn(language=afinn_lang, emoticons=emoticons)

    log("[afinn] scoring titles...")
    scores = titles_df["title"].map(af.score).astype(float)

    nonempty = (titles_df["title"].str.strip() != "")
    hits = (scores != 0.0) & nonempty
    cov_total = int(hits.sum())
    cov_share = cov_total / max(int(nonempty.sum()), 1)
    log(f"[afinn] nonempty_titles={int(nonempty.sum()):,} "
        f"with_nonzero_score={cov_total:,} "
        f"coverage={cov_share:.3f}")

    tmp = titles_df[["month"]].copy()
    tmp["score"] = scores
    tmp["nonempty"] = nonempty
    tmp["hit"] = hits

    ag = tmp.groupby("month").agg(
        sentiment_sum=("score", "sum"),
        sentiment_mean=("score", "mean"),
        sentiment_median=("score", "median"),
        titles_total=("score", "size"),
        titles_nonempty=("nonempty", "sum"),
        titles_with_hits=("hit", "sum"),
    ).reset_index().sort_values("month")

    ag["hit_coverage"] = ag["titles_with_hits"] / ag["titles_nonempty"].replace(0, np.nan)
    log(f"[afinn] months scored={ag.shape[0]}")
    return ag


def pearson_r(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]; b = b[mask]
    if a.size < 2:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])

def plot_dual_axis(
    df, month_col, y_left, y_right, out_png, title=None,
    left_color="tab:blue", right_color="tab:orange", line_width=1.8
):
    """
    Plot dual-axis time series: left y-axis for `y_left`, right y-axis for `y_right`.
    """
    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(111)

    x = pd.to_datetime(df[month_col] + "-01", errors="coerce")

    l1, = ax1.plot(x, df[y_left], color=left_color, linewidth=line_width,
                   label=y_left)
    ax1.set_xlabel("Month")
    ax1.set_ylabel(y_left, color=left_color)
    ax1.tick_params(axis='y', labelcolor=left_color)

    ax2 = ax1.twinx()
    l2, = ax2.plot(x, df[y_right], color=right_color, linewidth=line_width,
                   alpha=0.9, label=y_right)
    ax2.set_ylabel(y_right, color=right_color)
    ax2.tick_params(axis='y', labelcolor=right_color)

    ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    xmin, xmax = x.min(), x.max()
    ax1.set_xlim(xmin, xmax)

    for label in ax1.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")

    ax1.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)

    lines = [l1, l2]
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="upper left", frameon=True)

    ax1.set_title("Monthly Sentiment score vs nutrition intake")

    fig.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    log(f"[write] {out_png}")


def dump_example_months(titles_df, outdir, months_csv):
    """
    Save & briefly print the 'single dataframe per month' for demonstration.
    """
    sel = [m.strip() for m in months_csv.split(",") if m.strip()]
    for m in sel:
        ex = titles_df.loc[titles_df["month"] == m, ["month", "thread_id", "title"]]
        out = os.path.join(outdir, f"monthly_titles_{m}.csv")
        ex.to_csv(out, index=False)
        log(f"[examples] month={m} rows={len(ex):,} -> {out}")
        # Print first 5 rows
        with pd.option_context("display.max_colwidth", 120):
            print(ex.head(5).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(
        description="Q2: Monthly sentiment (AFINN) vs nutrient correlation & plot."
    )
    ap.add_argument("--base", default="/projappl/project_2015109/work/Q1_outputs",
                    help="Directory containing monthly_titles.parquet and monthly_nutrients.parquet.")
    ap.add_argument("--titles", default=None,
                    help="Explicit path to monthly_titles.parquet (optional).")
    ap.add_argument("--nutrients", default=None,
                    help="Explicit path to monthly_nutrients.parquet (optional).")
    ap.add_argument("--nutrient", action="append", default=None,
                    help="Nutrient column to compare. Repeatable. Use 'ALL' to run ENERC_kcal, PROT, FAT, CHOAVL.")
    ap.add_argument("--afinn-lang", default="fi",
                    help="AFINN language code (default: fi).")
    ap.add_argument("--emoticons", action="store_true",
                    help="Include AFINN emoticon lexicon in scoring.")
    ap.add_argument("--outdir", default="/projappl/project_2015109/work/results_q2",
                    help="Output directory for artifacts.")
    ap.add_argument("--sent-stat",
                    choices=["sentiment_mean", "sentiment_sum", "sentiment_median"],
                    default="sentiment_mean",
                    help="Which monthly sentiment aggregate to use.")
    ap.add_argument("--example-months", default="2001-01,2005-03",
                    help="Comma-separated months to dump as example monthly dataframes.")

    ap.add_argument("--left-color", default="tab:blue",
                    help="Line color for sentiment series.")
    ap.add_argument("--right-color", default="tab:orange",
                    help="Line color for nutrient series.")
    ap.add_argument("--line-width", type=float, default=1.8,
                    help="Line width for both series.")
    args = ap.parse_args()

    args.outdir = os.path.join(args.outdir, args.sent_stat)
    os.makedirs(args.outdir, exist_ok=True)
    log(f"[info] Using output folder: {args.outdir}")

    if args.titles:
        titles_df = pd.read_parquet(args.titles)
        log(f"[read] {args.titles}")
    else:
        titles_df = load_monthly_titles(args.base)

    if args.nutrients:
        nutrients_df = pd.read_parquet(args.nutrients)
        log(f"[read] {args.nutrients}")
    else:
        nutrients_df = load_monthly_nutrients(args.base)

    dump_example_months(titles_df, args.outdir, args.example_months)

    month_sent = score_titles_monthly(
        titles_df, afinn_lang=args.afinn_lang, emoticons=args.emoticons
    )

    sent_path_parquet = os.path.join(args.outdir, "monthly_sentiment.parquet")
    sent_path_csv = os.path.join(args.outdir, "monthly_sentiment.csv")
    month_sent.to_parquet(sent_path_parquet, index=False)
    month_sent.to_csv(sent_path_csv, index=False)
    log(f"[write] {sent_path_parquet}")
    log(f"[write] {sent_path_csv}")

    merged = pd.merge(nutrients_df, month_sent, on="month", how="inner")
    merged_path = os.path.join(args.outdir, "monthly_sentiment_vs_nutrients.parquet")
    merged.to_parquet(merged_path, index=False)
    log(f"[write] {merged_path}")

    if args.nutrient is None or "ALL" in [n.upper() for n in args.nutrient]:
        nutrients_to_run = ["ENERC_kcal", "PROT", "FAT", "CHOAVL"]
    else:
        nutrients_to_run = args.nutrient

    sent_col = args.sent_stat
    if sent_col not in merged.columns:
        raise SystemExit(f"Sentiment column '{sent_col}' not in merged table.")

    summary_path = os.path.join(args.outdir, "q2_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as fout:
        fout.write(f"Sentiment column: {sent_col}\n")
        fout.write(f"AFINN language: {args.afinn_lang} (emoticons={args.emoticons})\n")
        cov = month_sent["hit_coverage"].mean(skipna=True)
        fout.write(f"Mean monthly AFINN hit coverage (non-empty titles): {cov:.6f}\n")

    for nutrient in nutrients_to_run:
        if nutrient not in merged.columns:
            log(f"[warn] Nutrient '{nutrient}' not found in table. Skipping.")
            continue

        r = pearson_r(merged[sent_col], merged[nutrient])
        log(f"[corr] Pearson r between {sent_col} and {nutrient}: {r:.4f}")

        with open(summary_path, "a", encoding="utf-8") as fout:
            fout.write(f"Nutrient: {nutrient} | Pearson r: {r:.6f}\n")

        png = os.path.join(args.outdir, f"sentiment_vs_{nutrient}.png")
        title = f"Monthly {sent_col} vs {nutrient}"
        plot_dual_axis(
            merged.sort_values("month"),
            "month",
            sent_col,
            nutrient,
            png,
            title=title,
            left_color=args.left_color,
            right_color=args.right_color,
            line_width=args.line_width,
        )

    log("[done] Q2 sentiment pipeline complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"[error] {e}")
        raise

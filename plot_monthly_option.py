import os, sys, argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

INP = "/projappl/project_2015109/work/processed/monthly_nutrients.parquet"
OUT = "/projappl/project_2015109/work/results"
os.makedirs(OUT, exist_ok=True)

BROWN = "#8B4513"

def log(m): 
    """
    Simple log to stdout
    """
    print(m, flush=True)

def set_quarterly_month_ticks(ax, dates):
    """
    Put x-axis ticks at Jan, Apr, Jul, Oct across all years present in `dates`.
    """
    dates = pd.to_datetime(dates)
    want = [1, 4, 7, 10]
    tick_locs = dates[dates.dt.month.isin(want)]
    ax.set_xticks(tick_locs)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")

def set_dense_month_ticks(ax):
    """
    Set dense month ticks on the x-axis.
    """
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")

def plot_series(dates, y, title, ylabel, outp):
    """
    Plot a time series.
    """
    plt.figure(figsize=(20, 5))
    ax = plt.gca()

    if len(y) == 1:
        ax.scatter(dates, y, color=BROWN)
    else:
        ax.plot(dates, y, color=BROWN, linewidth=1.6)

    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel(ylabel)

    if len(dates) <= 14:
        set_dense_month_ticks(ax)
    else:
        set_quarterly_month_ticks(ax, dates)

    ax.grid(alpha=0.2, linestyle="--", linewidth=0.6)
    plt.tight_layout()
    plt.savefig(outp, dpi=150)
    plt.close()
    log(f"[write] {outp}")

def parse_args():
    """
    Parse command-line arguments.
    """
    p = argparse.ArgumentParser(description="Plot monthly nutrient series")
    p.add_argument("--cols", nargs="+", help="Specific nutrient columns to plot (e.g. ENERC_kcal PROT)")
    p.add_argument("--year", type=int, help="Plot only this year (e.g. 2012)")
    p.add_argument("--month", type=str, help="Plot only this month YYYY-MM (e.g. 2012-05)")
    p.add_argument("--from", dest="date_from", type=str, help="Filter start (YYYY-MM)")
    p.add_argument("--to", dest="date_to", type=str, help="Filter end (YYYY-MM)")
    return p.parse_args()

def apply_filters(df, args):
    """
    Apply filters to the dataframe.
    """
    df = df.copy()
    df["month"] = pd.to_datetime(df["month"])

    if args.year:
        df = df[df["month"].dt.year == args.year]
        log(f"[filter] year={args.year} -> rows={len(df)}")

    if args.month:
        wanted = pd.to_datetime(args.month + "-01")
        df = df[df["month"] == wanted]
        log(f"[filter] month={args.month} -> rows={len(df)}")

    if args.date_from:
        df = df[df["month"] >= pd.to_datetime(args.date_from + "-01")]
    if args.date_to:
        end = (pd.to_datetime(args.date_to + "-01") + pd.offsets.MonthBegin(1))
        df = df[df["month"] < end]
    if args.date_from or args.date_to:
        log(f"[filter] range {args.date_from or '...'}..{args.date_to or '...'} -> rows={len(df)}")

    return df.sort_values("month")

def main():
    """
    Main entry point.
    """
    args = parse_args()
    log(f"[info] reading {INP}")
    df = pd.read_parquet(INP)
    if df.empty:
        log("[error] monthly_nutrients is empty")
        sys.exit(1)

    df = apply_filters(df, args)
    if df.empty:
        log("[warn] after filtering, nothing to plot")
        sys.exit(0)

    if args.cols:
        series = [c for c in args.cols if c in df.columns]
        if not series:
            log(f"[error] none of --cols are in dataframe: {args.cols}")
            sys.exit(1)
        log(f"[info] plotting {series}")
    else:
        preferred = ["ENERC_kcal", "PROT", "FAT", "CHOAVL"]
        series = [c for c in preferred if c in df.columns]
        if not series:
            series = [c for c in df.columns if c not in ("month", "food_count")][:4]
        log(f"[info] plotting {series}")

    tag = []
    if args.year: tag.append(str(args.year))
    if args.month: tag.append(args.month)
    if args.date_from or args.date_to: tag.append(f"{args.date_from or ''}-{args.date_to or ''}")
    tag = "_".join([t for t in tag if t]) or "all"

    for col in series:
        outp = f"{OUT}/monthly_{col}_{tag}.png"
        plot_series(
            df["month"],
            df[col],
            title=f"Monthly nutrition intake - {col}",
            ylabel=col,
            outp=outp,
        )

    if "food_count" in df.columns:
        outp = f"{OUT}/monthly_food_count_{tag}.png"
        plot_series(
            df["month"],
            df["food_count"],
            title="Monthly distinct food count",
            ylabel="food_count",
            outp=outp,
        )

if __name__ == "__main__":
    main()

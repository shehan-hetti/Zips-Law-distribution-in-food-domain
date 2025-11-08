import os, json, glob, argparse, sys
import pandas as pd
from collections import defaultdict

THREADS_DIR = "/projappl/project_2015109/work/processed/s24_year_threads"
FINELI_PARQ = "/projappl/project_2015109/work/processed/fineli_food_nutrients.parquet"
OUTDIR = "/projappl/project_2015109/work/processed"
os.makedirs(OUTDIR, exist_ok=True)

def log(msg):
    """
    Simple log to stdout
    """
    print(msg, flush=True)

def main(years=None, dedupe=True):
    """
    Aggregate monthly food and nutrient data from Suomi24 forum threads matched
    with Fineli nutrient records
    """
    paths = sorted(glob.glob(f"{THREADS_DIR}/threads_*.jsonl"))
    if years:
        years = set(str(y) for y in years)
        paths = [p for p in paths if any(p.endswith(f"_{y}.jsonl") for y in years)]
    if not paths:
        log(f"[error] No thread files found in {THREADS_DIR} (years={years if years else 'ALL'})")
        sys.exit(1)

    log(f"[info] Loading Fineli nutrients: {FINELI_PARQ}")
    nut = pd.read_parquet(FINELI_PARQ).set_index("FOODID")
    log(f"[info] Nutrient table shape: {nut.shape}. Example cols: {list(nut.columns)[:10]}")

    month_foods = defaultdict(set)
    titles_rows = []

    total_lines = 0
    total_threads = 0
    skipped_dupe_lines = 0

    seen_thread_month = set()

    for path in paths:
        log(f"[read] {path}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                total_lines += 1
                rec = json.loads(line)

                m = rec.get("month")
                tid = rec.get("thread_id")

                if not m or not tid:
                    if m and rec.get("title"):
                        titles_rows.append({"month": m, "thread_id": tid or "", "title": rec["title"]})
                    continue

                key = (tid, m)
                if dedupe and key in seen_thread_month:
                    skipped_dupe_lines += 1
                    continue
                seen_thread_month.add(key)

                total_threads += 1

                for fid in rec.get("food_ids", []):
                    try:
                        month_foods[m].add(int(fid))
                    except Exception:
                        pass

                if rec.get("title"):
                    titles_rows.append({"month": m, "thread_id": tid, "title": rec["title"]})

    log(f"[info] read lines: {total_lines:,}, threads seen: {total_threads:,}, months with foods: {len(month_foods)}")
    if dedupe:
        log(f"[info] de-dup skipped lines (duplicate (thread_id, month)): {skipped_dupe_lines:,}")
    log(f"[info] titles collected: {len(titles_rows):,}")

    titles = pd.DataFrame(titles_rows)
    if not titles.empty:
        titles = titles.drop_duplicates(subset=["month", "thread_id", "title"])
    titles_out = f"{OUTDIR}/monthly_titles.parquet"
    titles.to_parquet(titles_out, index=False)
    log(f"[write] {titles_out}  (rows={len(titles)})")

    rows = []
    for m, fids in sorted(month_foods.items()):
        if not fids:
            continue

        sub = nut.reindex(list(fids)).dropna(how="all")
        sums = sub.sum(numeric_only=True)
        row = {"month": m, "food_count": len(fids)}
        row.update(sums.to_dict())
        rows.append(row)

    month_df = pd.DataFrame(rows).sort_values("month")

    if "ENERC" in month_df.columns:
        month_df["ENERC_kcal"] = month_df["ENERC"] / 4.184

    outp = f"{OUTDIR}/monthly_nutrients.parquet"
    month_df.to_parquet(outp, index=False)
    log(f"[write] {outp}  (rows={len(month_df)}, cols={len(month_df.columns)})")
    if len(month_df):
        log(f"[sample]\n{month_df.head(20)}")
    else:
        log("[warn] monthly_nutrients is empty. Did any threads contain foods?")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="*", help="List of years to include (e.g., 2010 2011). Default: all found.")
    ap.add_argument("--no-dedupe", action="store_true", help="Disable (thread_id, month) de-duplication.")
    args = ap.parse_args()
    yrs = [int(y) for y in args.years] if args.years else None
    main(yrs, dedupe=not args.no_dedupe)

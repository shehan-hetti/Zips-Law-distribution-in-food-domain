import argparse, json, logging, math, os, sys, re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.stem.snowball import SnowballStemmer
    HAVE_NLTK = True
except Exception:
    HAVE_NLTK = False

def setup_logger(log_dir: Path, name="q7_zipf"):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    fh = logging.FileHandler(log_path); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger, log_path

FINNISH_REGEX = re.compile(r"[a-zåäö]+", re.IGNORECASE)

def tokenize_regex(text: str):
    return FINNISH_REGEX.findall(str(text).lower())

def tokenize_nltk(text: str):
    if not HAVE_NLTK:
        return tokenize_regex(text)
    from nltk.tokenize import word_tokenize
    return [t.lower() for t in word_tokenize(str(text)) if any(ch.isalpha() for ch in t)]

def build_preprocessor(use_stopwords: bool, stem: str, logger):
    stops = set()
    stemmer = None

    if use_stopwords:
        if HAVE_NLTK:
            try:
                stops = set(nltk_stopwords.words("finnish"))
            except LookupError:
                logger.info("[warn] NLTK finnish stopwords not found; continuing without stopwords.")
                stops = set()
        else:
            logger.info("[warn] NLTK not available; continuing without stopwords.")
    if stem and stem.lower() == "finnish":
        if HAVE_NLTK:
            stemmer = SnowballStemmer("finnish")
        else:
            logger.info("[warn] NLTK not available; stemming disabled.")

    def normalize(tokens):
        if use_stopwords and stops:
            tokens = [t for t in tokens if t not in stops]
        if stemmer:
            tokens = [stemmer.stem(t) for t in tokens]
        return tokens
    return normalize

def ols_loglog(x_rank, y_freq):
    x = np.log(x_rank.astype(float))
    y = np.log(y_freq.astype(float))

    A = np.vstack([np.ones_like(x), x]).T
    coeff, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coeff
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    n = len(x); k = 1
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k - 1) if n > (k + 1) else float("nan")
    return {
        "a": float(coeff[0]),
        "b": float(coeff[1]),
        "R2": r2,
        "adj_R2": adj_r2,
        "n": int(n)
    }, yhat

def main():
    ap = argparse.ArgumentParser(description="Q7: Zipf fit for thread-title length bins.")
    ap.add_argument("--base", required=True, help="Project base dir (e.g., /projappl/.../work)")
    ap.add_argument("--tokenizer", choices=["regex","nltk"], default="regex")
    ap.add_argument("--use-stopwords", action="store_true", help="Remove Finnish stopwords (if available)")
    ap.add_argument("--stem", choices=["none","finnish"], default="none")
    ap.add_argument("--binning", choices=["equal","quantile"], default="equal")
    ap.add_argument("--bins", type=int, default=20)
    args = ap.parse_args()

    base = Path(args.base)
    results = base / "results_q7"
    results.mkdir(parents=True, exist_ok=True)
    logs = base / "logs"
    logger, log_path = setup_logger(logs, "q7_zipf_titles")

    logger.info(f"[start] base={base}")
    logger.info(f"[param] tokenizer={args.tokenizer} stopwords={args.use_stopwords} stem={args.stem} binning={args.binning} bins={args.bins}")
    logger.info(f"[log] writing to {log_path}")

    titles_pq = base / "processed" / "monthly_titles.parquet"
    if not titles_pq.exists():
        logger.error(f"[error] missing {titles_pq}")
        sys.exit(2)

    tok_fn = tokenize_regex if args.tokenizer == "regex" else tokenize_nltk
    normalize = build_preprocessor(args.use_stopwords, args.stem, logger)

    logger.info(f"[read] {titles_pq}")
    df = pd.read_parquet(titles_pq).dropna(subset=["title"])

    def length_from_title(s: str) -> int:
        toks = tok_fn(s)
        toks = normalize(toks)
        return len(toks)

    logger.info("[compute] title lengths")
    df["len"] = df["title"].astype(str).map(length_from_title)

    out_lengths = results / "title_lengths.parquet"
    df_out = df[["month","thread_id","len"]].copy()
    df_out.to_parquet(out_lengths, index=False)
    logger.info(f"[write] {out_lengths}")

    logger.info(f"[bin] method={args.binning} bins={args.bins}")
    if args.binning == "equal":
        bins = pd.cut(df["len"], bins=args.bins, include_lowest=True)
    else:
        q = np.linspace(0, 1, args.bins + 1)
        edges = np.unique(df["len"].quantile(q).values)
        if len(edges) < (args.bins + 1):
            logger.info(f"[warn] duplicate quantile edges reduced bins to {len(edges)-1}")
        bins = pd.cut(df["len"], bins=edges, include_lowest=True, duplicates="drop")

    freq = bins.value_counts().sort_values(ascending=False).reset_index()
    freq.columns = ["bin","freq"]
    freq["rank"] = np.arange(1, len(freq) + 1)

    lefts, rights = [], []
    for cat in freq["bin"]:
        if pd.isna(cat):
            lefts.append(np.nan); rights.append(np.nan)
        else:
            lefts.append(float(cat.left)); rights.append(float(cat.right))
    freq["left"] = lefts; freq["right"] = rights

    by_range = freq.sort_values(["left","right"]).reset_index(drop=True)
    by_range["bin"] = by_range["bin"].astype(str)
    out_bins = results / "length_bins.parquet"
    by_range.to_parquet(out_bins, index=False)
    logger.info(f"[write] {out_bins}")

    positive = freq[freq["freq"] > 0].copy()
    dropped = len(freq) - len(positive)
    if dropped > 0:
        logger.info(f"[warn] dropped {dropped} zero-frequency bins before log-log fit")
    positive = positive.sort_values("rank")
    positive["log_rank"] = np.log(positive["rank"].astype(float))
    positive["log_freq"] = np.log(positive["freq"].astype(float))

    fit, yhat = ols_loglog(positive["rank"].values, positive["freq"].values)
    slope_b = fit["b"]
    zipf_s = -slope_b
    intercept_a = fit["a"]

    positive["fit_line"] = intercept_a + slope_b * positive["log_rank"]
    out_fit = results / "zipf_freq_vs_rank.parquet"
    positive[["rank","freq","log_rank","log_freq","fit_line","left","right"]].to_parquet(out_fit, index=False)
    logger.info(f"[write] {out_fit}")

    fig = plt.figure(figsize=(8,6), dpi=150)
    ax = plt.gca()
    ax.scatter(positive["rank"], positive["freq"], s=35, label="Bins (freq by rank)")

    xx = np.linspace(positive["rank"].min(), positive["rank"].max(), 200)
    yy = np.exp(intercept_a + slope_b * np.log(xx))
    ax.plot(xx, yy, label=f"Fit: f = C * r^{slope_b:.3f}", linewidth=2)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Rank r (by bin frequency)")
    ax.set_ylabel("Frequency f")
    ax.set_title("Zipf fit of title-length bins")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    txt = f"R²={fit['R2']:.4f}, adj.R²={fit['adj_R2']:.4f}\nZipf exponent s≈{-slope_b:.3f}"
    ax.text(0.03, 0.03, txt, transform=ax.transAxes)

    out_png = results / "zipf_freq_vs_rank.png"
    plt.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    logger.info(f"[write] {out_png}")

    summary = {
        "tokenizer": args.tokenizer,
        "use_stopwords": bool(args.use_stopwords),
        "stem": args.stem,
        "binning": args.binning,
        "bins_requested": int(args.bins),
        "bins_after_drop_zeros": int(len(positive)),
        "zipf_log_model": "log(freq) = a + b*log(rank)",
        "a": intercept_a,
        "b": slope_b,
        "zipf_exponent_s": zipf_s,
        "R2": fit["R2"],
        "adj_R2": fit["adj_R2"],
        "n_points_fit": fit["n"]
    }
    (results / "q7_summary.txt").write_text(
        "Zipf fit on title-length bins\n"
        f"Tokenizer: {args.tokenizer}\n"
        f"Stopwords removed: {bool(args.use_stopwords)}\n"
        f"Stemming: {args.stem}\n"
        f"Binning: {args.binning} ({args.bins} bins requested)\n"
        f"Bins used in fit (freq>0): {len(positive)}\n"
        f"a = {intercept_a:.6f}\n"
        f"b = {slope_b:.6f}  (Zipf exponent s ≈ {-slope_b:.6f})\n"
        f"R^2 = {fit['R2']:.6f}\n"
        f"Adjusted R^2 = {fit['adj_R2']:.6f}\n"
    )
    with (results / "q7_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"[write] {results/'q7_summary.txt'}")
    logger.info("[done] Q7 Zipf pipeline complete.")

if __name__ == "__main__":
    main()

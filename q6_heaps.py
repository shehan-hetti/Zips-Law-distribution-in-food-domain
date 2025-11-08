import argparse, json, math, os, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, RegexpTokenizer
    from nltk.stem import SnowballStemmer
    _NLTK_OK = True
except Exception:
    _NLTK_OK = False

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def build_tokenizer(kind: str):
    """
    kind: 'regex' (default) or 'nltk'
    - regex keeps a-z åäö (lower/upper), splits on anything else.
    """
    if kind == "nltk" and _NLTK_OK:
        try:
            import nltk
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt", quiet=True)
            try:
                nltk.data.find("tokenizers/punkt_tab")
            except Exception:
                pass
            def tok(text: str):
                return [t for t in word_tokenize(text.lower()) if t.isalpha()]
            return tok, "nltk.word_tokenize"
        except Exception:
            pass

    if _NLTK_OK:
        rtok = RegexpTokenizer(r"[A-Za-zÅÄÖåäö]+")
        def tok(text: str):
            return rtok.tokenize(text.lower())
        return tok, "nltk.RegexpTokenizer"
    else:
        import re
        rx = re.compile(r"[A-Za-zÅÄÖåäö]+")
        def tok(text: str):
            return rx.findall(text.lower())
        return tok, "regex"

def build_stopwords(enable: bool, extra: list[str]):
    if not enable:
        return set()
    base = set()
    if _NLTK_OK:
        try:
            from nltk.corpus import stopwords
            try:
                stopwords.words("finnish")
            except LookupError:
                nltk.download("stopwords", quiet=True)
            base = set(stopwords.words("finnish"))
        except Exception:
            base = set()

    base |= set(t.lower() for t in extra or [])
    return base

def build_stemmer(kind: str):
    """
    kind: 'none' or 'finnish'
    """
    if kind.lower() == "finnish" and _NLTK_OK:
        try:
            return SnowballStemmer("finnish").stem
        except Exception:
            pass
    return None

def fit_heaps(N, V):
    """
    Fit log(V) = a + b*log(N); return K=exp(a), beta=b, R2, adj_R2
    """
    x = np.log(N.astype(float))
    y = np.log(V.astype(float))

    x_mean, y_mean = x.mean(), y.mean()
    cov = ((x - x_mean) * (y - y_mean)).sum()
    var = ((x - x_mean) ** 2).sum()
    if var == 0:
        return dict(K=np.nan, beta=np.nan, R2=np.nan, adj_R2=np.nan)

    beta = cov / var
    a = y_mean - beta * x_mean
    y_hat = a + beta * x
    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = ((y - y_mean) ** 2).sum()
    R2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    n = len(x)
    p = 1
    adj_R2 = 1.0 - (1.0 - R2) * (n - 1) / (n - p - 1) if n > (p + 1) and not math.isnan(R2) else np.nan
    K = float(np.exp(a))
    return dict(K=K, beta=float(beta), R2=float(R2), adj_R2=float(adj_R2))

def main():
    ap = argparse.ArgumentParser(description="Q6: Heaps' law over cumulative monthly titles.")
    ap.add_argument("--base", required=True, help="Base project folder (e.g., /projappl/project_2015109/work)")
    ap.add_argument("--tokenizer", choices=["regex","nltk"], default="regex", help="Tokenizer choice (default: regex)")
    ap.add_argument("--use-stopwords", action="store_true", help="Remove Finnish stopwords (NLTK)")
    ap.add_argument("--stem", choices=["none","finnish"], default="none", help="Apply stemming (Snowball, Finnish)")
    ap.add_argument("--outdir", default="results_q6", help="Output subdir under base (default: results_q6)")
    ap.add_argument("--save-csv", action="store_true", help="Also save cumulative table as CSV")
    args = ap.parse_args()

    base = Path(args.base).resolve()
    outdir = base / args.outdir
    ensure_dir(outdir)

    log(f"[start] base={base}")
    log(f"[param] tokenizer={args.tokenizer} stopwords={args.use_stopwords} stem={args.stem}")

    p_titles = base / "processed" / "monthly_titles.parquet"
    if not p_titles.exists():
        log(f"[error] missing {p_titles}")
        sys.exit(1)

    tokenize, tok_name = build_tokenizer(args.tokenizer)
    sw = build_stopwords(args.use_stopwords, extra=["quot", "nbsp"])
    stem_fn = build_stemmer(args.stem)

    log(f"[prep] tokenizer={tok_name} stopwords={len(sw)} stemmer={'yes' if stem_fn else 'no'}")

    log(f"[read] {p_titles}")
    df = pd.read_parquet(p_titles).dropna(subset=["title"])
    df = df.sort_values("month")
    months = df["month"].unique().tolist()
    log(f"[months] count={len(months)} range={months[0]} -> {months[-1]}")

    vocab = set()
    cumN = 0
    rows = []
    for m in months:
        g = df.loc[df["month"]==m, "title"].astype(str)
        toks = []
        for t in g:
            ts = tokenize(t)
            if sw:
                ts = [w for w in ts if w not in sw]
            if stem_fn:
                ts = [stem_fn(w) for w in ts]
            toks.extend(ts)
        cumN += len(toks)
        vocab.update(toks)
        rows.append({"month": m, "N_tokens": cumN, "V_types": len(vocab)})
    cum = pd.DataFrame(rows).sort_values("month")

    ok = cum["N_tokens"].is_monotonic_increasing and cum["V_types"].is_monotonic_increasing
    log(f"[cumulative] rows={len(cum)} monotonic={ok}")

    fit = fit_heaps(cum["N_tokens"], cum["V_types"])
    log(f"[fit] K={fit['K']:.4g} beta={fit['beta']:.4f} R2={fit['R2']:.4f} adj_R2={fit['adj_R2']:.4f}")

    p_parq = outdir / "heaps_cumulative.parquet"
    cum.to_parquet(p_parq, index=False)
    log(f"[write] {p_parq}")
    if args.save_csv:
        p_csv = outdir / "heaps_cumulative.csv"
        cum.to_csv(p_csv, index=False)
        log(f"[write] {p_csv}")

    fit_obj = {
        "tokenizer": tok_name,
        "use_stopwords": bool(args.use_stopwords),
        "stem": args.stem,
        "K": fit["K"],
        "beta": fit["beta"],
        "R2": fit["R2"],
        "adj_R2": fit["adj_R2"],
        "n_points": int(len(cum)),
        "first_month": months[0],
        "last_month": months[-1],
    }
    with open(outdir / "heaps_fit.json", "w", encoding="utf-8") as f:
        json.dump(fit_obj, f, ensure_ascii=False, indent=2)
    with open(outdir / "q6_summary.txt", "w", encoding="utf-8") as f:
        f.write(
            "Heaps' law fit (V = K * N^beta)\n"
            f"Tokenizer: {tok_name}\n"
            f"Stopwords removed: {bool(args.use_stopwords)}\n"
            f"Stemming: {args.stem}\n"
            f"Points: {len(cum)} ({months[0]} .. {months[-1]})\n"
            f"K = {fit['K']:.6g}\n"
            f"beta = {fit['beta']:.6f}\n"
            f"R^2 = {fit['R2']:.6f}\n"
            f"Adjusted R^2 = {fit['adj_R2']:.6f}\n"
        )

    p_png = outdir / "heaps_loglog.png"
    X = cum["N_tokens"].astype(float).values
    Y = cum["V_types"].astype(float).values
    K = fit["K"]; beta = fit["beta"]
    x_line = np.linspace(X.min(), X.max(), 500)
    y_line = K * (x_line ** beta) if not (np.isnan(K) or np.isnan(beta)) else None

    plt.figure(figsize=(8,6))
    plt.loglog(X, Y, marker='o', linestyle='', alpha=0.5, label="Data (cumulative months)")
    if y_line is not None:
        plt.loglog(x_line, y_line, linewidth=2, label=f"Fit: V = {K:.3g} * N^{beta:.3f}")
    plt.xlabel("Tokens N (cumulative)")
    plt.ylabel("Vocabulary V (cumulative)")
    plt.title("Heaps’ law on Suomi24 monthly thread titles (cumulative)")
    plt.legend()
    txt = f"R²={fit['R2']:.3f}, adj.R²={fit['adj_R2']:.3f}"
    plt.annotate(txt, xy=(0.03, 0.05), xycoords='axes fraction')
    plt.tight_layout()
    plt.savefig(p_png, dpi=150)
    plt.close()
    log(f"[write] {p_png}")

    log("[done] Q6 Heaps-law pipeline complete.")

if __name__ == "__main__":
    main()

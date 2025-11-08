import os
import sys
import json
import subprocess
import re
import io
import shlex
import argparse
from collections import defaultdict

ZIP = "/scratch/project_2015109/data/suomi24/suomi24-2001-2017-vrt-v1-3.zip"
INDIR = "suomi24-2001-2017-vrt-v1-3/data"
OUTDIR = "/projappl/project_2015109/work/processed/s24_year_threads"
LEX = "/projappl/project_2015109/work/processed/fineli_lexicon.json"

os.makedirs(OUTDIR, exist_ok=True)

def parse_attrs(tag_line: str) -> dict:
    """
    Parse attributes from a <text ...> line into a dict.
    """
    attrs = {}
    for m in re.finditer(r'(\w+)="([^"]*)"', tag_line):
        attrs[m.group(1)] = m.group(2)
    return attrs

def month_from(attrs: dict) -> str | None:
    """
    Prefer thread_start_datetime (YYYY-MM-DD HH:MM:SS), fall back to date (YYYY-MM-DD).
    Returns YYYY-MM or None.
    """
    dt = attrs.get("thread_start_datetime") or attrs.get("date")
    if not dt or len(dt) < 7:
        return None
    return dt[:7]

def load_lexicon(path: str):
    """
    Load matching lexicon produced by fineli_prep.py.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    lex = {tuple(k.split()): set(v) for k, v in raw.items()}
    max_n = max((len(k) for k in lex.keys()), default=1)
    return lex, max_n

def find_foods_in_lemmas(lemmas, lex, max_n):
    """
    Longest-first n-gram scan over lemmas using the lexicon.
    Returns a set of FOODIDs.
    """
    found = set()
    lemmas_length = len(lemmas)
    i = 0
    while i < lemmas_length:
        matched_any = False
        for n in range(min(max_n, lemmas_length - i), 0, -1):
            ngram = tuple(lemmas[i:i + n])
            if ngram in lex:
                found.update(lex[ngram])
                i += n
                matched_any = True
                break
        if not matched_any:
            i += 1
    return found

def open_year_stream(zip_path: str, year: str, verbose: bool = False):
    """
    Open a tolerant text stream for a given year's VRT via 'unzip -p'.
    Wraps the BYTES pipe with a UTF-8 decoder that replaces invalid bytes,
    don't crash on rare non-UTF8 artifacts.
    """
    inner = f"{INDIR}/s24_{year}.vrt"
    cmd = f'unzip -p {shlex.quote(zip_path)} {shlex.quote(inner)}'
    if verbose:
        print(f"[info] unzip command: {cmd}", flush=True)
    p = subprocess.Popen(
        cmd, shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    text_stream = io.TextIOWrapper(
        p.stdout, encoding="utf-8", errors="replace", newline=""
    )
    return p, text_stream

def process_year(year: str,
                 zip_path: str = ZIP,
                 outdir: str = OUTDIR,
                 lex_path: str = LEX,
                 limit_texts: int | None = None,
                 log_every: int = 10000,
                 write_empty: bool = True,
                 verbose: bool = False):
    """ 
    Process a single year of Suomi24 data.
    """

    lex, max_n = load_lexicon(lex_path)
    if verbose:
        print(f"[info] Lexicon loaded: {len(lex)} n-grams, max_n={max_n}", flush=True)

    out_path = os.path.join(outdir, f"threads_{year}.jsonl")

    threads_foods = defaultdict(set)
    threads_month = {}
    threads_title = {}

    month_foods = defaultdict(set)

    n_texts = 0
    n_token_lines = 0
    n_threads = 0
    n_threads_with_foods = 0
    n_titles = 0
    months_seen = set()

    current_attrs = None
    buffer_lemmas = []

    p, stream = open_year_stream(zip_path, year, verbose=verbose)
    had_bad_bytes = False

    try:
        for line in stream:
            if "�" in line:
                had_bad_bytes = True

            line = line.rstrip("\n")

            if line.startswith("<text "):
                if current_attrs is not None and buffer_lemmas:
                    fids = find_foods_in_lemmas(buffer_lemmas, lex, max_n)
                    if fids:
                        tid0 = current_attrs.get("thread_id")
                        if tid0:
                            before_sz = len(threads_foods[tid0])
                            threads_foods[tid0].update(fids)
                            if before_sz == 0 and len(threads_foods[tid0]) > 0:
                                n_threads_with_foods += 1
                            m0 = threads_month.get(tid0)
                            if m0:
                                month_foods[m0].update(fids)
                buffer_lemmas = []

                current_attrs = parse_attrs(line)
                n_texts += 1

                tid = current_attrs.get("thread_id")
                if tid and tid not in threads_month:
                    n_threads += 1
                    m = month_from(current_attrs)
                    if m:
                        threads_month[tid] = m
                        months_seen.add(m)

                if tid and current_attrs.get("msg_type") == "thread_start":
                    t = current_attrs.get("title_orig") or current_attrs.get("title")
                    if t and tid not in threads_title:
                        threads_title[tid] = t
                        n_titles += 1

                if log_every and (n_texts % log_every == 0):
                    latest_month = max(months_seen) if months_seen else "-"
                    latest_count = len(month_foods.get(latest_month, set()))
                    print(f"[progress] year={year} texts={n_texts:,} "
                          f"tokens={n_token_lines:,} threads={n_threads:,} "
                          f"threads_with_foods={n_threads_with_foods:,} months={len(months_seen)} "
                          f"month_foods[{latest_month}]={latest_count:,}",
                          flush=True)

                if limit_texts and n_texts >= limit_texts:
                    if verbose:
                        print(f"[info] limit_texts reached: {limit_texts}, stopping stream.", flush=True)
                    break

            elif line.startswith("</text"):
                if current_attrs is not None and buffer_lemmas:
                    fids = find_foods_in_lemmas(buffer_lemmas, lex, max_n)
                    if fids:
                        tid1 = current_attrs.get("thread_id")
                        if tid1:
                            before_sz = len(threads_foods[tid1])
                            threads_foods[tid1].update(fids)
                            if before_sz == 0 and len(threads_foods[tid1]) > 0:
                                n_threads_with_foods += 1
                            m1 = threads_month.get(tid1)
                            if m1:
                                month_foods[m1].update(fids)
                current_attrs = None
                buffer_lemmas = []

            elif line and not line.startswith("<"):
                parts = line.split("\t")
                if len(parts) >= 3:
                    lemma = parts[2].lower()
                    lemma = re.sub(r"[^\wåäöæøéáíóúüß\-]+", "", lemma)
                    if lemma:
                        buffer_lemmas.append(lemma)
                n_token_lines += 1

    finally:
        try:
            stream.detach()
        except Exception:
            pass
        rc = p.wait()
        if rc != 0:
            err = p.stderr.read().decode("utf-8", "replace")
            raise RuntimeError(f"unzip pipeline failed (rc={rc}). stderr:\n{err}")

    wrote_any = False
    with open(out_path, "w", encoding="utf-8") as fout:
        for tid in threads_month.keys():
            rec = {
                "thread_id": tid,
                "month": threads_month.get(tid),
                "title": threads_title.get(tid),
                "food_ids": sorted(list(threads_foods.get(tid, set())))
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            wrote_any = True

    if had_bad_bytes:
        print(f"[warn] {year}: encountered invalid UTF-8 bytes; replaced with U+FFFD", file=sys.stderr, flush=True)

    if not wrote_any and not write_empty and os.path.exists(out_path):
        os.remove(out_path)
        out_msg = "(no records; file removed due to --no-write-empty)"
    else:
        out_msg = out_path

    print(f"[done] year={year} texts={n_texts:,} tokens={n_token_lines:,} "
          f"threads={n_threads:,} threads_with_foods={n_threads_with_foods:,} "
          f"months={len(months_seen)} out={out_msg}", flush=True)

    if months_seen:
        summary = " ".join(f"{m}:{len(month_foods.get(m, set())):,}" for m in sorted(months_seen))
        print(f"[month_food_counts] {summary}", flush=True)

def main():
    ap = argparse.ArgumentParser(description="Stream-process a Suomi24 VRT year into per-thread food mentions.")
    ap.add_argument("year", help="Year like 2010")
    ap.add_argument("--zip", dest="zip_path", default=ZIP, help=f"Path to ZIP (default: {ZIP})")
    ap.add_argument("--outdir", default=OUTDIR, help=f"Output dir (default: {OUTDIR})")
    ap.add_argument("--lex", dest="lex_path", default=LEX, help=f"Lexicon JSON (default: {LEX})")
    ap.add_argument("--limit-texts", type=int, default=None, help="Stop after N <text> blocks (debug/sanity).")
    ap.add_argument("--log-every", type=int, default=10000, help="Log every N <text> blocks (0 to disable).")
    ap.add_argument("--no-write-empty", action="store_true", help="Remove output if no records.")
    ap.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    args = ap.parse_args()

    process_year(
        year=str(args.year),
        zip_path=args.zip_path,
        outdir=args.outdir,
        lex_path=args.lex_path,
        limit_texts=args.limit_texts,
        log_every=args.log_every,
        write_empty=(not args.no_write_empty),
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()

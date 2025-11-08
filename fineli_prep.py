import pandas as pd
import json, re, os, sys
from collections import defaultdict, Counter

try:
    import nltk
    from nltk.corpus import stopwords
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

BASE = "/scratch/project_2015109/data/fineli/fineli_extracted/utf8"
OUTDIR = "/projappl/project_2015109/work/processed"
os.makedirs(OUTDIR, exist_ok=True)

def log(msg: str):
    """
    Simple log to stdout
    """
    print(msg, flush=True)

def norm_text(s: str) -> str:
    """
    Normalize text by lowercasing and removing unwanted characters.
    """
    s = s.lower()
    s = re.sub(r"[^\wåäöæøéáíóúüß\- ]+", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(s: str):
    """
    Tokenize text.
    """
    return [t for t in re.split(r"\s+", s) if t]

def count_by_ngram_len(lexicon: dict) -> dict:
    """
    Count the number of keys in the lexicon by their n-gram length.
    """
    return dict(Counter(len(k) for k in lexicon.keys()))

log(f"[info] Reading component table: {BASE}/component.csv")
comp = pd.read_csv(f"{BASE}/component.csv", sep=";")

log(f"[info] Reading component values: {BASE}/component_value.csv")
vals = pd.read_csv(f"{BASE}/component_value.csv", sep=";", decimal=",")

log(f"[info] Reading Finnish food names: {BASE}/foodname_FI.csv")
names = pd.read_csv(f"{BASE}/foodname_FI.csv", sep=";")

unit_map = dict(zip(comp["EUFDNAME"], comp["COMPUNIT"]))

log("[info] Pivoting nutrient values to wide table")
wide = vals.pivot(index="FOODID", columns="EUFDNAME", values="BESTLOC").fillna(0.0).reset_index()

log("[info] Merging Finnish food names")
foods = names[names["LANG"] == "FI"][["FOODID", "FOODNAME"]].drop_duplicates()
wide = foods.merge(wide, on="FOODID", how="left")

nutrients_out = f"{OUTDIR}/fineli_food_nutrients.parquet"
wide.to_parquet(nutrients_out, index=False)
log(f"[write] {nutrients_out}  (rows={len(wide)}, cols={len(wide.columns)})")

# Build lexicon from Finnish names
# Start with 1-3 grams
# Then prune noisy unigrams and “connector” ngrams
log("[info] Building initial 1-3 gram lexicon from food names")
lexicon = defaultdict(set)  # key: ngram tuple -> set(food_ids)

for row in foods.itertuples(index=False):
    fid = int(row.FOODID)
    name = norm_text(str(row.FOODNAME))
    toks = tokenize(name)
    for n in (1, 2, 3):
        if len(toks) < n:
            continue
        for i in range(len(toks) - n + 1):
            ngram = tuple(toks[i : i + n])
            if all(len(t) >= 2 for t in ngram):
                lexicon[ngram].add(fid)

pre_counts = count_by_ngram_len(lexicon)
log(f"[lexicon][pre] total keys: {len(lexicon)}  by_len: {pre_counts}")

#  Use NLTK Finnish stopwords + Fineli-specific
try:
    FINN_STOP = set(stopwords.words("finnish"))
except Exception as e:
    print(f"[warn] Could not load NLTK Finnish stopwords ({e}); using minimal fallback list")
    FINN_STOP = {"ja", "tai", "sekä", "on", "ei", "uusi"}

FINN_STOP.update({"ug", "vl", "arc"})

AMBIG_LIMIT = 8
SAFE_UNIGRAMS = set()

to_delete = []

# Drop multi-grams that include connector words only (not all stopwords)
CONNECTOR_STOP = {"ja", "tai", "sekä", "&"}
for key in list(lexicon.keys()):
    if len(key) >= 2 and any(tok in CONNECTOR_STOP for tok in key):
        to_delete.append(key)

for k in to_delete:
    lexicon.pop(k, None)
if to_delete:
    log(f"[lexicon][prune] removed {len(to_delete)} multi-gram keys containing stop-connectors")

# 1-3) Prune unigrams
to_delete = []
for key, ids in lexicon.items():
    if len(key) != 1:
        continue
    tok = key[0]

    if tok in FINN_STOP:
        to_delete.append(key)
        continue
    # too short or not alphabetic
    if len(tok) < 3 or not re.match(r"^[a-zåäö\-]+$", tok):
        to_delete.append(key)
        continue

    if tok not in SAFE_UNIGRAMS and len(ids) > AMBIG_LIMIT:
        to_delete.append(key)

for k in to_delete:
    lexicon.pop(k, None)

post_counts = count_by_ngram_len(lexicon)
log(f"[lexicon][post] total keys: {len(lexicon)}  by_len: {post_counts}")


uni_sample = [(k[0], len(v)) for k, v in lexicon.items() if len(k) == 1]
uni_sample.sort(key=lambda kv: -kv[1])
if uni_sample:
    log("[lexicon] top remaining unigrams (token -> #FOODIDs):")
    for tok, n in uni_sample[:20]:
        log(f"  {tok:20s} {n}")
else:
    log("[lexicon] no unigrams remain (all matching via 2–3 grams)")


lex_out = { " ".join(k): sorted(list(v)) for k, v in lexicon.items() }
lex_path = f"{OUTDIR}/fineli_lexicon.json"
with open(lex_path, "w", encoding="utf-8") as f:
    json.dump(lex_out, f, ensure_ascii=False)
log(f"[write] {lex_path}")

units_path = f"{OUTDIR}/fineli_component_units.json"
with open(units_path, "w", encoding="utf-8") as f:
    json.dump(unit_map, f, ensure_ascii=False)
log(f"[write] {units_path}")

log("Saved:\n" + "\n".join([nutrients_out, lex_path, units_path]))

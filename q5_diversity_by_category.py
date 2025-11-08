import argparse, os, sys, json, glob, math
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def log(msg):
    print(f"[{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def build_rules():

    return [
        ("Hedelmät & marjat", [
            "OMENA","BANAANI","APPELSIINI","SITRUUNA","HEDELMÄ","MARJA","MUSTIKKA","PUOLUKKA","MANSIKKA","APRIKKOOSI","APRIKOOSI",
            "PERSIKKA","ANANAS","MELONI","CANTALOUPE","VERKKOMELONI","KARPALO","MUSTAHERUKKA","PUNAHERUKKA","KARVIAINEN","VADELMA",
            "LAKKA","SUOMUURAIN","HILLA","MANDARIINI","GREIPPI","KIIVI","KIRSIKKA","LUUMU","AVOKADO","PÄÄRYNÄ","RUSINA","VIIKUNA",
            "TAATELI","GUAVA","PERSIMON","SHARON","KAKI","PAPAIJA","MANGO","JUOLUKKA","SHARON, KAKI, PERSIMON"
        ]),
        ("Vihannekset & juurekset", [
            "PORKKANA","PERUNA","SIPULI","TOMAATTI","KURKKU","SALAATTI","KAALI","JUURES","PAPRIKA","VALKOSIPULI","LANTTU","NAURIS",
            "PUNAJUURI","RETIISI","PALSTERNAKKA","MAA-ARTISOKKA","MUSTAJUURI","PERSILJA","NOKKONEN","RAPARPERI","SELLERI","VARSISELLERI",
            "JUURISELLERI","LEHTISELLERI","PARSA","SIENI","TATTI","HAPERO","HERKKUSIENI","KANTARELLI","KORVASIENI","KURPITSA","MANGOLDI",
            "VIHANNESKRASSI","TILLI","(ARC)","JUURIKASVIS","LEHTIKASVIS","ITU","SINIMAILASEN","BAMBUNVERSO","CHILI","BASILIKA","PINAATTI",
            "KANGASROUSKU","BATAATTI","LATVA-ARTISOKKA","HAAPAROUSKU","SUPPILOVAHVERO","KARVAROUSKU","KORIANTERI","JUUREKSET",
            "KUUSENKERKKÄ","VUOHENPUTKI","VÄINÖNPUTKI","LATVA-ARTISOKKA, KEITETTY","BATAATTI, KUORITTU","BATAATTI, UUNISSA",
            "MERILEVÄ","MERILEVA","LEVÄ","LEVA","NORI","WAKAME","KOMBU"
        ]),
        ("Viljat & viljatuotteet", [
            "LEIPÄ","JAUHO","HIUTALE","MAKARONI","PASTA","RIISI","PUURO","COUSCOUS","KUSKUS","RUIS","VEHNÄ","OHRA","KAURA","GRAHAM",
            "MANNASUURIMO","SEMOLIINA","MAISSI","MALLAS","TACOKUORI","VOHVELI","VOHVELIKUORI","HAPANKORPPU","MURO","MYSLI",
            "POPCORN","MIKROPOPCORN","KRUTONKI"
        ]),
        ("Liha & lihatuotteet", [
            "LIHA","NAUTA","SIKA","KANA","BROILERI","MAKKARA","JAUHELIHA","LEIKKELE","SIAN","NAUDAN","LAMPAAN","JÄNIS","SELKÄSILAVA",
            "KYLKI","LAPA","POTKA","RINTA","FILEE","ULKOFILEE","PAISTI","KANI","KALKKUNA","METSÄLINTU","PORO","MAKSA","MUNUAINEN",
            "VERI","PEKONI","PORSAANKYLJYS","PORSAANLEIKE","RIISTA","NAKKI","PAISTETTU","GRILLATTU","LEIVITETTY","MEETVURSTI",
            "TARTARPIHVI","PORSAS","RIIS-","NUGGETTI","NYHTÖPOSSU","PULLED PORK","HÄRKIS-NUUDELIWOKKI","KASVISPYÖRYKKÄ, SAARIOINEN",
            "KASVISPIHVI","SAARIOINEN"
        ]),
        ("Kala & merenelävät", [
            "KALA","LOHI","SILAKKA","MUIKKU","SILLI","SEITI","TONNIKALA","KATKARAPU","AHVEN","HAUKI","LAHNA","SIIKA","MADE","KUHA",
            "TURSKA","KAMPELA","ANKERIAS","TAIMEN","NAHKIAINEN","ANJOVIS","PUNA-AHVEN","SINIPALLAS","MÄTI","MÄTITAHNA","SIMPUKKA",
            "SINISIMPUKKA","KAVIAARI","SAMMEN","SAVUSTETTU","RAPU","OSTERI","ETANA","HUMMERI","SÄRKI","TILAPIA","KILOHAILI"
        ]),
        ("Maito & maitotuotteet", [
            "MAITO","JUUSTO","JOGURT","JOGURTTI","KERMA","VIILI","RAEJUUSTO","PIIMÄ","HERAJAUHE","ÄIDINMAIDONKORVIKE","RAHKA",
            "VALKUAISJAUHE","KELTUAISJAUHE","MIFU","HERAPROTEIINI","HERAPROTEIINIKONSENTRAATTI"
        ]),
        ("Munat", ["MUNA","KANANMUNA"]),
        ("Kastikkeet, liemet & mausteet", [
            "KASTIKE","LIEMI","LIHALIEMI","KALALIEMI","SOIJAKASTIKE","MAUSTE","SINAPPI","SUOLA","KETSUPPI","ETIKKA","MAJONEESI",
            "MUSTAPIPPURI","VALKOPIPPURI","KANELI","NEILIKKA","INKIVÄÄRI","KARDEMUMMA","OREGANO","TIMJAMI","PIPARJUURI",
            "RUOKASOODA","MAIZENA","SUURUSTE","YRTTI","KORIANTERI","UUTE"
        ]),
        ("Palkokasvit", ["PAPU","HERNE","LINSSI","SOIJA","TEMPEH","TOFU"]),
        ("Pähkinät & siemenet", ["PÄHKINÄ","MANTELI","SESAM","SIEMEN","KASTANJA"]),
        ("Rasvat & öljyt", ["ÖLJY","MARGARIINI","VOI","RASVA"]),
        ("Sokerit & makeiset", [
            "SOKERI","SIIRAPPI","HUNAJA","MAKEINEN","SUKLAA","KARKKI","LAKRITSI","TOFFEE","MARSIPAANI","KAAKAOJAUHE","KAAKAO","JOGURTTIRUSINA",
            "PURUKUMI","MARMELADI","STEVIA","JÄÄTELÖ","PIRTELÖ","CRÈME BRÛLÉE","DONITSI","MUNKKI","SALMIAKKI","PASTILLI"
        ]),
        ("Leivonnaiset & jälkiruoat", [
            "LEIVONNAI","KEKSI","KAKKU","PULLA","PATUKKA","PIIRAKKA","MÄMMI","PASHA","JOULUTORTTU","VANUKAS","KOHOKAS","MARENKI",
            "TÄYTEKAKUN","KÄÄRETORTTU","UNELMATORTTU","KÄÄRETORTUN POHJA","UNELMATORTUN POHJA","KAKUN KUORRUTUS","KREEMI",
            "RAHKAPIIRAKAN","RAHKATÄYTE","MOKKAKUORRUTUS","OHUKAINEN","PIIRAKAN RAHKATÄYTE","KAMPANISU"
        ]),
        ("Juomat", [
            "JUOMA","MEHU","LIMU","KAHVI","TEE","OLUT","VIINI","VIINA","VODKA","GINI","GIN","KONJAKKI","ROMMI","VISKI","LIKÖÖRI",
            "LONG DRINK","LONKERO","SIIDERI","SIMA","KIVENNÄISVESI","MINERAALIVESI","VESI","VESIJOHTOVESI","ROMMIKOLA","GIN TONIC",
            "TOM COLLINS","DRY MARTINI","BOOLI"
        ]),
        ("Valmisruoat & prosessoidut", [
            "VALMIS","PIZZA","HAMPURILAINEN","LASAGNE","KEITTO","RISOTTO","RATATOUILLE","PYTTIPANNU","ITALIANPATA","KASVISWOKKI",
            "WOKKIVIHANNEKSET","PINAATTIOHUKAINEN","KASVISKUSKUS","FETA-KASVISPANNU","KEVÄTKÄÄRYLE","KASVISPYÖRYKKÄ","KASVISPIHVI",
            "RÖSSYPOTTU","TOFU-NUUDELIWOKKI","NYHTÖPOSSU","NUGGETTI","PULLED PORK","SUSHI","KEBAB","ISKENDER","HÄRKIS-NUUDELIWOKKI",
            "ATERIANKORVIKE","CAMBRIDGE","WOKKI"
        ]),
        ("Lisäaineet & makeutusaineet", [
            "LEIVINJAUHE","GLUKOOSI","FRUKTOOSI","SORBITOLI","ASPARTAAMI","SAKARIINI","KSYLITOLI","LIIVATE","HIIVA","TIIVISTE",
            "TÄRKKELYS","MALTODEKSTRIINI","SYKLAMAATTI"
        ]),
    ]

def make_categorizer():
    rules = build_rules()
    def categorize(name: str) -> str:
        s = ("" if pd.isna(name) else str(name)).upper()
        for cat, keys in rules:
            if any(k in s for k in keys):
                return cat
        return "Muut / ei luokiteltu"
    return categorize

def month_ticks(ax, xlabels):
    if not len(xlabels): return
    idx = [i for i, lab in enumerate(xlabels) if lab.endswith(("-01","-04","-07","-10"))]
    ax.set_xticks(idx)
    ax.set_xticklabels([xlabels[i] for i in idx], rotation=45, ha="right")

def safe_pearson(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0,1])

def shannon_effective(counts: Counter):
    total = sum(counts.values())
    if total <= 0:
        return 0.0, 0.0
    H = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            H -= p * math.log(p)
    return H, math.exp(H)

def main():
    ap = argparse.ArgumentParser(description="Q5: Category-based food diversity vs nutrient intake (monthly & yearly).")
    ap.add_argument("--base", required=True, help="Work base path (e.g., /projappl/project_2015109/work)")
    ap.add_argument("--nutrient", default="ENERC_kcal", help="Nutrient column to compare (default ENERC_kcal)")
    ap.add_argument("--write-cat-csv", default=None, help="Optional CSV path to write FOODID,FOODNAME,category using final rules")
    ap.add_argument("--emit-counts", action="store_true", help="Include JSON composition column 'category_counts' in diversity parquets")
    ap.add_argument("--emit-counts-long", action="store_true", help="Also write tidy long-form composition parquets")
    args = ap.parse_args()

    base = args.base.rstrip("/")
    nutrient = args.nutrient
    emit_counts = args.emit_counts or args.emit_counts_long

    outdir = os.path.join(base, "results_q5")
    os.makedirs(outdir, exist_ok=True)

    threads_dir = os.path.join(base, "processed", "s24_year_threads")
    fineli_p = os.path.join(base, "processed", "fineli_food_nutrients.parquet")
    monthly_nutr_p = os.path.join(base, "Q1_outputs", "monthly_nutrients.parquet")

    log(f"[start] base={base}")
    log(f"[param] nutrient={nutrient}")
    log(f"[read] {fineli_p}")
    fineli = pd.read_parquet(fineli_p, columns=["FOODID","FOODNAME"]).dropna()
    categorize = make_categorizer()
    fineli["category"] = fineli["FOODNAME"].map(categorize)
    cat_map = dict(zip(fineli["FOODID"], fineli["category"]))
    coverage = (fineli["category"] != "Muut / ei luokiteltu").mean()
    log(f"[coverage] category mapping coverage={coverage:.3f}")

    if args.write_cat_csv:
        out_csv = args.write_cat_csv
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        fineli[["FOODID","FOODNAME","category"]].to_csv(out_csv, index=False)
        log(f"[write] category mapping CSV -> {out_csv}")

    monthly_sets = defaultdict(lambda: {"cat_set": set(), "food_mentions": 0, "lines": 0})
    monthly_counts = defaultdict(lambda: Counter())
    total_lines = 0

    for pth in sorted(glob.glob(os.path.join(threads_dir, "threads_20*.jsonl"))):
        log(f"[read] {pth}")
        with open(pth, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                m = r.get("month")
                if not m:
                    continue
                ids = r.get("food_ids") or []
                if ids:
                    cats = [cat_map.get(fid, "Muut / ei luokiteltu") for fid in ids]
                    for c in cats:
                        if c != "Muut / ei luokiteltu":
                            monthly_sets[m]["cat_set"].add(c)
                            monthly_counts[m][c] += 1
                monthly_sets[m]["food_mentions"] += len(ids)
                monthly_sets[m]["lines"] += 1
                total_lines += 1
    log(f"[scan] lines processed: {total_lines:,}")

    m_rows = []
    for m in sorted(monthly_sets):
        d = monthly_sets[m]
        H, effN = shannon_effective(monthly_counts[m])
        row = {
            "month": m,
            "categories_distinct": len(d["cat_set"]),
            "category_entropy": H,
            "category_effN": effN,
            "food_mentions": d["food_mentions"],
            "lines": d["lines"],
        }
        if emit_counts:
            row["category_counts"] = json.dumps(monthly_counts[m], ensure_ascii=False)
        m_rows.append(row)
    monthly_div = pd.DataFrame(m_rows)
    log(f"[monthly] rows={len(monthly_div)} months={monthly_div['month'].nunique()}")

    m_div_pq = os.path.join(outdir, "monthly_category_diversity.parquet")
    monthly_div.to_parquet(m_div_pq, index=False)
    log(f"[write] {m_div_pq}")

    if args.emit_counts_long:
        long_rows = []
        for m, cnt in monthly_counts.items():
            for c, v in cnt.items():
                long_rows.append({"month": m, "category": c, "count": int(v)})
        monthly_long = pd.DataFrame(long_rows)
        m_long_pq = os.path.join(outdir, "monthly_category_composition.parquet")
        monthly_long.to_parquet(m_long_pq, index=False)
        log(f"[write] {m_long_pq}")

    log(f"[read] {monthly_nutr_p}")
    monthly_nutr = pd.read_parquet(monthly_nutr_p)
    if nutrient not in monthly_nutr.columns:
        log(f"[error] Nutrient '{nutrient}' not found in monthly_nutrients.parquet")
        sys.exit(2)
    monthly_merge = monthly_div.merge(monthly_nutr[["month", nutrient]], on="month", how="inner")
    mm_pq = os.path.join(outdir, "monthly_category_vs_nutrients.parquet")
    monthly_merge.to_parquet(mm_pq, index=False)
    log(f"[write] {mm_pq}")

    r_month_rich = safe_pearson(monthly_merge["categories_distinct"], monthly_merge[nutrient])
    r_month_effN = safe_pearson(monthly_merge["category_effN"], monthly_merge[nutrient])
    if np.isnan(r_month_rich):
        log("[corr] Monthly richness has zero variance; r set to NaN.")
    log(f"[corr] Pearson r (monthly) categories_distinct vs {nutrient}: {r_month_rich:.4f}")
    log(f"[corr] Pearson r (monthly) category_effN vs {nutrient}: {r_month_effN:.4f}")

    fig, ax1 = plt.subplots(figsize=(18,6))
    ax2 = ax1.twinx()
    x = list(range(len(monthly_merge)))

    color_left = "crimson"
    color_right = "mediumblue"

    ax1.plot(x, monthly_merge["category_effN"].values, color=color_left, label="Category diversity (effective N, monthly)", linewidth=1.8)
    ax2.plot(x, monthly_merge[nutrient].values, color=color_right, label=nutrient, linewidth=1.8)

    ax1.set_xlabel("Month")
    ax1.set_ylabel("Effective categories (exp(H))", color=color_left)
    ax2.set_ylabel(nutrient, color=color_right)

    ax1.tick_params(axis='y', labelcolor=color_left)
    ax2.tick_params(axis='y', labelcolor=color_right)

    ax1.grid(True, alpha=0.3)
    xlabels = monthly_merge["month"].tolist()
    month_ticks(ax1, xlabels)

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    ax1.set_title("Monthly Category diversity vs Nutrition intake")

    fig.tight_layout()
    m_png = os.path.join(outdir, f"monthly_categories_effN_vs_{nutrient}.png")
    fig.savefig(m_png, dpi=140)
    plt.close(fig)
    log(f"[write] {m_png}")

    yearly_sets = defaultdict(lambda: {"cat_set": set(), "food_mentions": 0, "lines": 0})
    yearly_counts = defaultdict(lambda: Counter())
    total_lines = 0

    for pth in sorted(glob.glob(os.path.join(threads_dir, "threads_20*.jsonl"))):
        log(f"[read] {pth}")
        with open(pth, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                m = r.get("month")
                if not m:
                    continue
                y = m[:4]
                ids = r.get("food_ids") or []
                if ids:
                    cats = [cat_map.get(fid, "Muut / ei luokiteltu") for fid in ids]
                    for c in cats:
                        if c != "Muut / ei luokiteltu":
                            yearly_sets[y]["cat_set"].add(c)
                            yearly_counts[y][c] += 1
                yearly_sets[y]["food_mentions"] += len(ids)
                yearly_sets[y]["lines"] += 1
                total_lines += 1
    log(f"[scan] lines processed: {total_lines:,}")

    y_rows = []
    for y in sorted(yearly_sets):
        d = yearly_sets[y]
        H, effN = shannon_effective(yearly_counts[y])
        row = {
            "year": y,
            "categories_distinct_year": len(d["cat_set"]),
            "category_entropy_year": H,
            "category_effN_year": effN,
            "food_mentions": d["food_mentions"],
            "lines": d["lines"],
        }
        if emit_counts:
            row["category_counts"] = json.dumps(yearly_counts[y], ensure_ascii=False)
        y_rows.append(row)
    yearly_div = pd.DataFrame(y_rows)
    log(f"[yearly] rows={len(yearly_div)} years={yearly_div['year'].nunique()}")

    y_div_pq = os.path.join(outdir, "yearly_category_diversity.parquet")
    yearly_div.to_parquet(y_div_pq, index=False)
    log(f"[write] {y_div_pq}")

    if args.emit_counts_long:
        long_rows = []
        for y, cnt in yearly_counts.items():
            for c, v in cnt.items():
                long_rows.append({"year": y, "category": c, "count": int(v)})
        yearly_long = pd.DataFrame(long_rows)
        y_long_pq = os.path.join(outdir, "yearly_category_composition.parquet")
        yearly_long.to_parquet(y_long_pq, index=False)
        log(f"[write] {y_long_pq}")

    log(f"[read] {monthly_nutr_p}")
    monthly_nutr = pd.read_parquet(monthly_nutr_p, columns=["month", nutrient])
    monthly_nutr["year"] = monthly_nutr["month"].str[:4]
    yearly_nutr = monthly_nutr.groupby("year", as_index=False)[nutrient].sum()
    y_merge = yearly_div.merge(yearly_nutr, on="year", how="inner")
    y_pq = os.path.join(outdir, "yearly_category_vs_nutrients.parquet")
    y_merge.to_parquet(y_pq, index=False)
    log(f"[write] {y_pq}")

    r_year_rich = safe_pearson(y_merge["categories_distinct_year"], y_merge[nutrient])
    r_year_effN = safe_pearson(y_merge["category_effN_year"], y_merge[nutrient])
    if np.isnan(r_year_rich):
        log("[corr] Yearly richness has zero variance; r set to NaN.")
    log(f"[corr] Pearson r (yearly) categories_distinct_year vs {nutrient}: {r_year_rich:.4f}")
    log(f"[corr] Pearson r (yearly) category_effN_year vs {nutrient}: {r_year_effN:.4f}")

    fig, ax1 = plt.subplots(figsize=(12,5))
    ax2 = ax1.twinx()
    x = list(range(len(y_merge)))

    color_left = "crimson"
    color_right = "mediumblue"

    ax1.plot(x, y_merge["category_effN_year"].values, color=color_left, label="Category diversity (effective N, yearly)", linewidth=1.8)
    ax2.plot(x, y_merge[nutrient].values, color=color_right, label=nutrient, linewidth=1.8)

    ax1.set_xlabel("Year")
    ax1.set_ylabel("Effective categories (exp(H))", color=color_left)
    ax2.set_ylabel(nutrient, color=color_right)

    ax1.tick_params(axis='y', labelcolor=color_left)
    ax2.tick_params(axis='y', labelcolor=color_right)

    ax1.set_xticks(x)
    ax1.set_xticklabels(y_merge["year"].tolist(), rotation=45, ha="right")

    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    ax1.set_title("Yearly Category diversity vs Nutrition intake")

    fig.tight_layout()
    y_png = os.path.join(outdir, f"yearly_categories_effN_vs_{nutrient}.png")
    fig.savefig(y_png, dpi=140)
    plt.close(fig)
    log(f"[write] {y_png}")

    summary = os.path.join(outdir, "q5_summary.txt")
    with open(summary, "w", encoding="utf-8") as f:
        f.write(f"Nutrient: {nutrient} | Pearson r (monthly; richness): {r_month_rich:.6f}\n")
        f.write(f"Nutrient: {nutrient} | Pearson r (monthly; effN): {r_month_effN:.6f}\n")
        f.write(f"Plot (monthly; effN): {m_png}\n")
        f.write(f"Nutrient: {nutrient} | Pearson r (yearly; richness): {r_year_rich:.6f}\n")
        f.write(f"Nutrient: {nutrient} | Pearson r (yearly; effN): {r_year_effN:.6f}\n")
        f.write(f"Plot (yearly; effN): {y_png}\n")
    log(f"[write] {summary}")
    log("[done] Q5 category-diversity pipeline complete.")

if __name__ == "__main__":
    main()

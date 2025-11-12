# 🧠 Zips Law Distribution in the Food Domain

This project explores linguistic and behavioral patterns in Finnish online food discussions using **Suomi24 forum data** combined with **Fineli food composition data**.  
It examines how language reflects social and nutritional behavior and tests whether classical linguistic scaling laws — **Heaps’ law** and **Zipf’s law** — hold true in this real-world domain.


## 📘 Overview

The analysis investigates how online conversations about food evolve over time, how sentiment and food diversity change, and how language complexity grows with data volume.  
Each stage corresponds to a specific research question in a sequence of tasks (Q1–Q8):

| Task | Description |
|------|--------------|
| **Q1** | Data preparation and alignment of Suomi24 and Fineli datasets. |
| **Q2** | Monthly sentiment analysis using the AFINN model. |
| **Q3** | Yearly sentiment trend visualization and correlation with nutritional intake. |
| **Q4** | Measurement of food diversity over time. |
| **Q5** | Manual categorization of food groups and correlation with nutrients. |
| **Q6** | Fitting **Heaps’ law** to study vocabulary growth and language scaling. |
| **Q7** | Fitting **Zipf’s law** to study word frequency and title-length distributions. |
| **Q8** | Literature reflection and evaluation of linguistic patterns in food-related language. |

## 🧩 Data Sources

- **Suomi24 Forum Dataset:** Finnish online discussion data (2001–2017).  
- **Fineli Database:** National food composition database by THL (Institute for Health and Welfare, Finland).

These datasets were processed into a unified structure containing thread titles, sentiment scores, nutritional indicators, and monthly aggregations.

## 🧠 Methods

- **Sentiment Analysis:** Lexicon-based (AFINN Finnish adaptation)
- **Food Categorization:** Manually designed mapping for Fineli items  
- **Linguistic Scaling:**
  - *Heaps’ Law:* \( V = K \times N^{\beta} \)
  - *Zipf’s Law:* \( f(r) \propto 1/r^s \)
- **Correlation Analysis:** Pearson’s r and adjusted R² metrics  
- **Visualization:** Matplotlib and Seaborn-based plots for all key trends

## 📊 Key Findings

- Vocabulary growth follows **Heaps’ law** with parameters  
  **K = 9.19**, **β = 0.709**, **R² ≈ 0.9999**
- Title length frequencies follow **Zipf’s law** with  
  **Exponent s ≈ 6.62**, **R² ≈ 0.977**
- Monthly correlation between category diversity (EffN) and energy intake (**ENERC_kcal**) was **r = 0.30–0.34**
- The results confirm that even informal online discussions exhibit strong linguistic regularities.

## 🧾 Requirements

- Python 3.12+
- Virtual environment (`venv`)
- Libraries:
  ```bash
  pandas
  numpy
  matplotlib
  seaborn
  pyarrow
  fastparquet
  nltk
  scipy

To install dependencies:
```
pip install -r requirements.txt
```
## 🚀 Running the Scripts

Each question has its own script under /scripts. Example run:
```
python scripts/q6_heaps.py \
  --base /projappl/project_2015109/work \
  --use-stopwords --stem finnish

```
Output files (e.g., plots, .parquet, .txt summaries) will be written under results_q* directories.

## 📈 Example Results

- results_q6/heaps_loglog.png → Heaps’ law log–log plot
- results_q7/zipf_freq_vs_rank.png → Zipf’s law frequency–rank plot
- results_q5/monthly_categories_effN_vs_ENERC_kcal.png → Correlation of category diversity and energy intake

## 💡 Usage Example
Run all scripts step by step to reproduce the pipeline:
```
# Q4 - Compute food diversity
python scripts/q4_diversity.py --base /projappl/project_2015109/work

# Q5 - Category-based diversity vs. nutrients
python scripts/q5_diversity_by_category.py --base /projappl/project_2015109/work --nutrient ENERC_kcal

# Q6 - Heaps’ Law fit
python scripts/q6_heaps.py --base /projappl/project_2015109/work --use-stopwords --stem finnish

# Q7 - Zipf’s Law fit
python scripts/q7_zipf_title_lengths.py --base /projappl/project_2015109/work --tokenizer regex --use-stopwords --stem finnish --binning equal --bins 20
```

## 🪄 License

This repository is released under the MIT License.
You are free to use, modify, and distribute with proper attribution.

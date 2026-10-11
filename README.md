# Goodreads Book Recommender with Spark

Our capstone for DS-GA 1004 Big Data at NYU (Spring 2026), by Adarsh Tiwari and Sheetal Sood.

We used the Goodreads interactions dataset (Wan & McAuley, RecSys 2018): about 223M user-book interactions across 876K users and 2.4M books. Everything ran on NYU's Dataproc cluster with PySpark. We built:

1. **Market segmentation:** MinHash LSH to find the 100 book pairs read by the most similar groups of users
2. **A popularity baseline**
3. **Three ALS recommenders:** one on explicit ratings only, one on implicit signals (read / reviewed) only, and one that combines both

The full story of what we ran, what broke and why is in [WALKTHROUGH.md](WALKTHROUGH.md). The original assignment is in [ASSIGNMENT.md](ASSIGNMENT.md).

## Results

Top-100 ranking metrics. The ALS models were evaluated on a 3% sample of users (about 22K) because of cluster limits. The popularity baseline was evaluated on the full split.

| Model | Signal | Val MAP@100 | Test MAP@100 | Test NDCG@100 |
| --- | --- | --- | --- | --- |
| Popularity | most-read books | 0.0311 | 0.0315 | 0.1036 |
| Explicit ALS | ratings only | 0.000012 | 0.000006 | 0.000059 |
| Implicit ALS | `is_read`, alpha = 40 | 0.0731 | 0.0738 | 0.2063 |
| Implicit ALS + reviews | `is_read` + 0.5 x `is_reviewed` | 0.0708 | 0.0719 | 0.2032 |
| **Combined ALS** | reads weighted by rating, plus reviews | **0.0734** | **0.0758** | **0.2089** |

What we took away from it:

- **Implicit feedback is what works here.** Implicit ALS gets about 2.3x the popularity baseline's MAP. Explicit ALS is near zero: only about 46% of interactions have a rating, and we evaluate on what people read, not what they rated. Even when we scored it only against rated books it stayed near zero.
- **Weighting reads by rating helped a little.** The combined model has the best numbers, but the gain over implicit ALS is small. Bootstrap 95% intervals on the 1% test sample (about 7.7K users, 500 resamples) overlap: implicit [0.0715, 0.0764], combined [0.0740, 0.0788]. So it's a consistent direction, not a decisive win.
- **The most "similar" books are mostly the same book.** The top MinHash pairs have Jaccard similarity of 0.896 to 0.950, but most are different editions of one title. After merging editions with union-find, the 100 pairs collapse to 37 distinct work pairs.

## What's in the repo

The main pipeline is at the top level, numbered by step:

| File | What it does |
| --- | --- |
| `step0_csv_to_parquet.py` | Converts the raw CSVs to Parquet and writes sanity counts |
| `step1_split.py` | Per-user 70/15/15 train/val/test split (seed 1004). Users with fewer than 10 interactions go to train only. Also makes a 1% sample for prototyping |
| `step2_minhash.py` | MinHash LSH over books with at least 200 readers, top-100 pairs |
| `step3_popularity.py` | Popularity baselines |
| `step4_als_explicit_*.py` | Explicit ALS: tuning grid (rank x regParam), training, evaluation |
| `step5_als_implicit_*.py` | Implicit ALS and the version with reviews added |
| `step6_als_combined_*.py` | Combined explicit + implicit ALS |
| `step_3pct_eval.py`, `step_d3_3pct_eval.py`, `step_d4v2_3pct_eval.py` | Final evaluations on the 3% user sample |
| `eval_helpers.py` | Shared ranking-metric code |

Other folders:

- `scripts/` has follow-up diagnostics: edition dedup, the rated-only evaluation, bootstrap intervals, and an implicit ALS tuning sweep we wrote but didn't get to run. See [scripts/README.md](scripts/README.md).
- `results/` and the `step*.txt` / `step*.csv` files are saved outputs from our runs.

## Running it

The scripts were written for NYU's Dataproc cluster (Spark on YARN with HDFS), and each one has its `spark-submit` command in the docstring at the top. They read and write under `hdfs:///user/<netid>_nyu_edu/goodreads`, and the NetID comes from a `NETID` environment variable:

```
export NETID=your_netid
```

That works for client mode. For scripts submitted in cluster mode, also pass it to the driver with `--conf spark.yarn.appMasterEnv.NETID=$NETID`.

The raw data isn't in this repo. The Goodreads dataset is available from the [UCSD Goodreads dataset page](https://mengtingwan.github.io/data/goodreads.html).

## Team

Adarsh Tiwari and Sheetal Sood. We made the design decisions together, and the commit history shows who pushed what.

## AI use

We used Claude (Anthropic) and Gemini (Google) for help with writing, debugging PySpark scripts and proofreading. This README was also put together with Claude's help from our walkthrough and results files. The experimental design, parameter choices and interpretation of results were our own.

## References

- Mengting Wan and Julian McAuley. Item Recommendation on Monotonic Behavior Chains. RecSys 2018.
- Yifan Hu, Yehuda Koren and Chris Volinsky. Collaborative Filtering for Implicit Feedback Datasets. ICDM 2008.

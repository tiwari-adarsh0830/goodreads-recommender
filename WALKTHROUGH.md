# Capstone Walkthrough — DSGA1004 Big Data, Spring 2026
**Team:** Adarsh Tiwari, Sheetal Sood

This is our project walkthrough of what we built, ran, and found. The README has the formal project description.

---

## The Data

The Goodreads dataset has ~223M user-book interactions across ~876K users and ~2.4M books. Each row is `user_id, book_id, is_read, rating, is_reviewed`. The three fields mean very different things:

- `rating` — explicit, the user actually scored the book 1–5. Only 45.7% of interactions have this.
- `is_read` — implicit, the user marked it as read. Doesn't mean they liked it.
- `is_reviewed` — implicit engagement, they wrote something. Also doesn't mean they liked it.

This distinction ended up being the most important thing in the whole project. More on that below.

First thing we did was convert the raw CSV to Parquet. The CSV is huge and slow to repeatedly scan — Parquet made every downstream job significantly faster.

---

## Data Split (step1)

Split was per-user, 70/15/15 train/val/test using a seeded random ordering of each user's interactions. Users with fewer than 10 interactions were kept in train only — you can't meaningfully evaluate recommendations for someone with almost no history.

- Total eligible users: 772,730
- Ineligible (train only): 103,415
- seed=1004 throughout for reproducibility

For prototyping we also created a small split — 1% sample of users (~9,026 users), same proportions. Final ALS evaluations use a 3% sample (~22,000 users) which the professor confirmed is acceptable given cluster constraints.

---

## D1 — Market Segmentation with MinHash/LSH (step2)

The goal was to find the 100 most similar book pairs by user overlap — books read by the same groups of people.

We used MinHash with LSH (Locality Sensitive Hashing) to approximate Jaccard similarity at scale. Computing exact Jaccard over 2.4M books is completely infeasible — MinHash lets you do it in linear time with a tunable accuracy tradeoff.

**Parameters:**
- `MIN_READERS = 200` — filtered out books with fewer than 200 readers. Books with tiny readerships produce noisy similarity estimates and aren't useful for market segmentation.
- `NUM_HASH_TABLES = 32`
- `LSH_THRESHOLD = 0.3`
- Books surviving filter: 67,100

**Results:** Top-100 pairs have Jaccard similarity ranging from 0.8963 to 0.9502. Full list in `step2_top100_pairs.csv`, sorted descending.

**Interesting finding:** The top pairs are almost entirely adjacent book IDs. These are different editions of the same work (hardcover vs paperback, different publishers, etc.). Goodreads treats each edition as a separate book ID, so the same readers show up on both. This is a property of how Goodreads structures its data, not a bug in the algorithm.

**Follow-up: Edition Deduplication.** We added a union-find deduplication pass. The original 100 pairs contained 62 edition edges. After union-find, 139 books collapsed into 86 work groups, and only 37 distinct work-pairs remained. The top exact Jaccard after deduplication was 0.9328. Output is in `results/top100_pairs_deduped.csv`.

---

## D2 — Popularity Baseline (step3)

Before building anything complicated, we needed a simple baseline to beat.

We tried two variants:

**POP_READ** — rank books by total number of users who marked them as read. Simple count, no rating involved.

**POP_BAYES** — Bayesian average rating, trying to balance rating score with number of ratings. This completely collapsed. A handful of niche books with ~25 perfect 5-star ratings dominated the entire ranking. The Bayesian prior wasn't strong enough to regularize against books with tiny but perfect rating histories. MAP@100 was essentially zero.

So POP_READ is the valid baseline. Evaluated on the full split (742K/745K users):

| Split | P@100 | MAP@100 | NDCG@100 | n users |
|---|---|---|---|---|
| VAL (full) | 0.0219 | 0.0311 | 0.1022 | 742,484 |
| TEST (full) | 0.0223 | 0.0315 | 0.1036 | 745,822 |

---

## D3 — Explicit ALS (step4)

First real model. ALS with `implicitPrefs=False`, trained only on rows where `rating > 0`.

**Hyperparameter tuning:** ran a 12-combination grid on the small split — rank ∈ {10, 20, 50, 100} × regParam ∈ {0.01, 0.1, 1.0}. Best: rank=50, regParam=0.1.

**Final params:** rank=50, regParam=0.1, maxIter=10, seed=1004

**Results (3% sample):**

| Split | P@100 | MAP@100 | NDCG@100 | n users |
|---|---|---|---|---|
| VAL (3%) | 0.000017 | 0.000012 | 0.000085 | 22,335 |
| TEST (3%) | 0.000016 | 0.000006 | 0.000059 | 22,418 |

Way below popularity baseline. This is the expected outcome, not a bug.

**Why:** The model trains on ~73M rated interactions (45.7% of all train rows). The ground truth for evaluation is `is_read=1` — which covers ALL consumption events, including the 54.3% of reads that were never rated. The model never saw those interactions. Explicit ALS optimizes rating prediction but we are evaluating on consumption coverage — a fundamental signal-target mismatch.

**Rated-only diagnostic:** We also evaluated on a rated-only ground truth (is_read=1 AND rating>0). Still effectively zero: P@100=5×10⁻⁶, MAP@100=1×10⁻⁶. Explicit ratings alone are too sparse for useful top-100 recommendation.

---

## D4 — Implicit ALS (step5) — Main Model

This is where things actually worked.

Switched to `implicitPrefs=True`. Instead of predicting a rating, the model learns from binary consumption signals using a confidence-weighted matrix factorization.

**Signal:** `confidence = 1 + alpha * is_read`
**alpha = 40.0** — chosen based on Hu, Koren, and Volinsky (2008) recommendation range.

**Training data:** ALL 159.8M train rows, not just rated ones. This is the key difference from D3.

**Final params:** rank=50, regParam=0.1, maxIter=10, alpha=40.0, seed=1004, implicitPrefs=True

**Results (3% sample):**

| Split | P@100 | MAP@100 | NDCG@100 | n users |
|---|---|---|---|---|
| VAL (3%) | 0.0465 | 0.0731 | 0.2040 | 22,524 |
| TEST (3%) | 0.0472 | 0.0738 | 0.2063 | 22,583 |

2.3× improvement in MAP over popularity baseline.

**D4v2 — adding is_reviewed: tried adding 0.5 * is_reviewed to the signal. Slightly hurt performance (MAP dropped from 0.0731 to 0.0708 on val).

---

## D5 — Combined Explicit + Implicit (step6) — Best Model

The idea: when a user has both read AND rated a book, the rating tells you how much they liked it. Pure `is_read` treats a 1-star read the same as a 5-star read.

**Signal:**
if rating > 0:
combined = (rating / 5.0) * is_read + 0.3 * is_reviewed
else:
combined = 0.5 * is_read + 0.3 * is_reviewed
Same ALS setup: implicitPrefs=True, alpha=40.0, rank=50, regParam=0.1.

**Results (3% sample):**

| Split | P@100 | MAP@100 | NDCG@100 | n users |
|---|---|---|---|---|
| VAL (3%) | 0.0466 | 0.0734 | 0.2042 | 21,616 |
| TEST (3%) | 0.0475 | 0.0758 | 0.2089 | 22,500 |

Best model across all metrics. Beats D4 on both val and test.

---

## Full Comparison

| Model | VAL MAP@100 | TEST MAP@100 | TEST NDCG@100 |
|---|---|---|---|
| D2 Popularity (POP_READ) | 0.0311 | 0.0315 | 0.1036 |
| D3 Explicit ALS | 0.000012 | 0.000006 | 0.000059 |
| D4 Implicit ALS | 0.0731 | 0.0738 | 0.2063 |
| D4v2 Implicit (+reviews) | 0.0708 | 0.0719 | 0.2032 |
| **D5 Combined (best)** | **0.0734** | **0.0758** | **0.2089** |

---

## Bootstrap Variance Bands

Because ALS evaluation uses a 3% user sample, we added bootstrap confidence intervals over users for D4 and D5 on the test split (B=500 resamples).

| Model | MAP@100 point | 95% CI |
|---|---|---|
| D4 Implicit ALS | 0.0738 | [0.0715, 0.0764] |
| D5 Combined ALS | 0.0758 | [0.0740, 0.0788] |

D5 remains best but intervals overlap — improvement is directionally consistent rather than statistically decisive on the 3% sample.

---

## Future Work

- Run full-split evaluation via checkpointed or user-sharded batches
- Run the D4 hyperparameter sweep (scripts/step5_als_implicit_tune.py is ready)
- Extend bootstrap CIs to D2 and D4v2
- Rerun MinHash with full candidate checkpoint for true deduplication
- Sensitivity sweep on D5 signal coefficients

---

## Contributions

This project was completed jointly by Adarsh Tiwari and Sheetal Sood. Every design decision was discussed together. Earlier commits came from Adarsh's machine and later ones from Sheetal's. Both members participated throughout.

## AI Disclosure

Claude (Anthropic) and Gemini (Google) were used for help with writing, debugging PySpark scripts, and proofreading. All experimental design decisions, parameter choices, and result interpretation were our own.

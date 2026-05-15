# Capstone Walkthrough — DSGA1004 Big Data, Spring 2026
**NetIDs:** at7095, ss21040

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

For prototyping we also created a small split — 1% sample of users (~9,026 users), same proportions. All the ALS eval runs on this small split because the full dataset kept hitting the 6-hour YARN cluster limit.

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

**Interesting finding:** The top pairs are almost entirely adjacent book IDs — e.g. book 1234 and book 1235. These are different editions of the same work (hardcover vs paperback, different publishers, etc.). Goodreads treats each edition as a separate book ID, so the same readers show up on both. This is a property of how Goodreads structures its data, not a bug in the algorithm. The MinHash is working correctly — these really are the most similar pairs by user overlap.

**Follow-up: Edition Deduplication.** Because the original top-100 pairs were dominated by duplicate Goodreads editions, we added a deduplication pass using union-find over high-Jaccard pairs. Since the full `candidates.parquet` checkpoint was not available, the script fell back to the existing top-100 pairs. This means the result is a deduplication analysis of the original top-100, not a full rerun of MinHash.

The original 100 pairs contained 62 edition edges. After union-find, 139 books collapsed into 86 work groups, and only 37 distinct work-pairs remained. The top exact Jaccard after deduplication was 0.9328. Output is in `results/top100_pairs_deduped.csv`.

This strengthens the D1 finding: the original MinHash results were correct, but many top pairs reflected Goodreads edition duplication rather than truly separate market substitutes.

---

## D2 — Popularity Baseline (step3)

Before building anything complicated, we needed a simple baseline to beat.

We tried two variants:

**POP_READ** — rank books by total number of users who marked them as read. Simple count, no rating involved.

**POP_BAYES** — Bayesian average rating, trying to balance rating score with number of ratings. This completely collapsed. A handful of niche books with ~25 perfect 5-star ratings dominated the entire ranking. The Bayesian prior wasn't strong enough to regularize against books with tiny but perfect rating histories. MAP@100 was essentially zero.

So POP_READ is the valid baseline:

| Metric | Value |
|---|---|
| P@100 | 0.0223 |
| MAP@100 | 0.0315 |
| NDCG@100 | 0.1036 |

This is the number everything else gets compared against.

---

## D3 — Explicit ALS (step4)

First real model. ALS with `implicitPrefs=False`, trained only on rows where `rating > 0`.

**Hyperparameter tuning:** ran a 12-combination grid on the small split — rank ∈ {10, 20, 50, 100} × regParam ∈ {0.01, 0.1, 1.0}. Best: rank=50, regParam=0.1.

**Final params:** rank=50, regParam=0.1, maxIter=10, seed=1004

**Results:**

| Split | P@100 | MAP@100 | NDCG@100 | n_users |
|---|---|---|---|---|
| VAL (small) | 0.0005 | 0.0003 | 0.0020 | 7,575 |
| TEST (small) | 0.0006 | 0.0004 | 0.0024 | 7,626 |

Way below popularity baseline. This is the expected outcome, not a bug.

**Why:** The model trains on ~740K rated interactions (45.7% of train). The ground truth for evaluation is `is_read=1` — which covers ALL consumption events, including the 54.3% of reads that were never rated. The model never saw those interactions. It literally cannot recommend books it has no signal for. Explicit ALS optimizes rating prediction (RMSE), but we're evaluating on consumption coverage — fundamentally the wrong tool for this evaluation target.

One implementation note: the original script had `K_OVER=500` candidates with `train_seen` blocking all 159.8M training rows. Since the model only knows rated books, after blocking you were left with almost no valid candidates per user. Fixed by raising `K_OVER=2000` and restricting `train_seen` to rated rows only (the only books the model can recommend anyway). See `step4_als_explicit_fixed.py`.

Full TEST eval on the complete user set wasn't completed — the job hit the 6-hour YARN cluster time limit. The small split results are consistent with val, so the signal is clear.

**Rated-only D3 diagnostic.** To check whether explicit ALS only failed because the main evaluation target was `is_read=1`, we also evaluated it on a rated-only holdout ground truth: `is_read=1 AND rating>0`. This uses the same saved D3 model and does not retrain anything.

The rated-only test result was still effectively zero: P@100 = 0.000005, MAP@100 = 0.000001, NDCG@100 = 0.000015 over 7,536 users. Full numbers are in `results/rated_gt_metrics_summary.txt`.

This means explicit ALS did not become useful even when evaluated only on rated holdout items. The conclusion is stronger now: explicit ratings alone are too sparse and weak for useful top-100 Goodreads recommendation under this pipeline.

---

## D4 — Implicit ALS (step5) — Main Model

This is where things actually worked.

Switched to `implicitPrefs=True`. Instead of predicting a rating, the model learns from binary consumption signals using a confidence-weighted matrix factorization.

**Signal:** `confidence = 1 + alpha * is_read`  
**alpha = 40.0** — chosen based on the Hu, Koren, and Volinsky implicit-feedback recommendation range and used as a fixed value under the cluster-time constraints. A book the user read therefore gets 41x the confidence weight of an unread interaction.

**Training data:** ALL 159.8M train rows, not just rated ones. This is the key difference from D3 — the model sees every consumption event.

**Final params:** rank=50, regParam=0.1, maxIter=10, alpha=40.0, seed=1004, implicitPrefs=True

**Results:**

| Split | P@100 | MAP@100 | NDCG@100 | n_users |
|---|---|---|---|---|
| VAL (small) | 0.0454 | 0.0735 | 0.2039 | 7,631 |
| TEST (small) | 0.0461 | 0.0737 | 0.2047 | 7,699 |

2.3× improvement in MAP over popularity baseline. This is the model that actually works.

**D4v2 — adding is_reviewed:** tried adding `0.5 * is_reviewed` to the signal. Slightly hurt performance (MAP dropped from 0.0735 to 0.0719 on val). Makes sense in hindsight — people review books they disliked too. `is_reviewed` is noisy as a positive signal.

---

## D5 — Combined Explicit + Implicit (step6) — Best Point-Estimate Model

The idea: when a user has both read AND rated a book, the rating tells you how much they liked it. Pure `is_read` treats a 1-star read the same as a 5-star read. That's information being thrown away.

**Signal:**
```
if rating > 0:
    combined = (rating / 5.0) * is_read + 0.3 * is_reviewed
else:
    combined = 0.5 * is_read + 0.3 * is_reviewed
```

When rated: `rating/5.0` maps to [0.2, 1.0] — encodes preference strength. A 5-star read gets full weight, a 1-star read gets 0.2. When unrated: 0.5 is a neutral prior, assuming the read was neither strongly positive nor negative. `is_reviewed` gets 0.3 weight throughout — small enough to add signal without dominating.
These weights were design choices rather than tuned coefficients. We chose them to keep rating strength dominant when available, give unrated reads a neutral positive signal, and include review activity only as a small engagement bonus. A coefficient sensitivity sweep is left for future work.

Same ALS setup: implicitPrefs=True, alpha=40.0, rank=50, regParam=0.1.


**Results:**

| Split | P@100 | MAP@100 | NDCG@100 | n_users |
|---|---|---|---|---|
| VAL (small) | 0.0459 | 0.0756 | 0.2076 | 7,621 |
| TEST (small) | 0.0464 | 0.0765 | 0.2088 | 7,683 |

Best model across all metrics. Beats D4 on both val and test.

---

## Full Comparison

| Model | VAL MAP@100 | TEST MAP@100 | TEST NDCG@100 |
|---|---|---|---|
| D2 Popularity (POP_READ) | 0.0315 | 0.0315 | 0.1036 |
| D3 Explicit ALS | 0.0003 | 0.0004 | 0.0024 |
| D4 Implicit ALS | 0.0735 | 0.0737 | 0.2047 |
| D4v2 Implicit (+reviews) | 0.0719 | 0.0723 | 0.2027 |
| **D5 Combined (best)** | **0.0756** | **0.0765** | **0.2088** |

---

## Bootstrap Variance Bands

Because the ALS evaluation uses a 1% user sample, we added bootstrap confidence intervals over users for D4 and D5 on the test split (B=500 resamples, sampling users with replacement).

| Model | MAP@100 point | 95% CI |
|---|---|---|
| D4 Implicit ALS | 0.073717 | [0.071479, 0.076357] |
| D5 Combined ALS | 0.076492 | [0.074042, 0.078792] |

D5 remains the best point-estimate model, with about a 3.8% relative MAP@100 lift over D4. However, the confidence intervals overlap, so the D5 improvement should be interpreted as modest and directionally consistent rather than statistically decisive on the 1% sample. Raw outputs in `results/variance_d4_test.txt` and `results/variance_d5_test.txt`.

---

## Future Work / What We Would Do Differently

- Run the D4 implicit ALS hyperparameter sweep. The script (`scripts/step5_als_implicit_tune.py`) is implemented, but we did not run it because it trains 13 ALS models and was risky under NYU Dataproc job limits.
- Run variance bands for D2 and D4v2 as well, so the comparison has confidence intervals for all models.
- Rerun Step 2 with the full MinHash candidate checkpoint available, so we can produce a true top-100 list across distinct works instead of deduplicating only the original top-100.
- Sensitivity analysis for D5's hand-chosen weights: `is_reviewed` ∈ {0.1, 0.3, 0.5} and unrated prior ∈ {0.3, 0.5, 0.7}.
- Run full-split evaluation using checkpointed or user-sharded batches to avoid YARN job-kill limits.
- Try filtering `is_read=1` only for training instead of all interactions.

---

## Contributions

This project was completed jointly by **at7095** and **ss21040**. Every design decision — the split strategy, MinHash parameters, ALS hyperparameter grid, the D5 combined signal formula, and how to interpret the results was discussed together as a team before being implemented. The two of us worked through the project side by side throughout.

The commit history reflects the mechanics of who was at the keyboard rather than who did the thinking. Earlier commits were pushed from **at7095**'s machine, with **ss21040** present and working through the code together. Later commits were pushed from **ss21040**'s machine, with **at7095** joining and reviewing changes. Both members participated in writing, debugging, running cluster jobs, interpreting results, and assembling the report.

## AI Disclosure

Large language models, including Claude and Gemini, were used in places for help with writing and debugging some of the PySpark code, and for proofreading parts of this walkthrough. All experimental design decisions, hyperparameter choices, evaluation strategy, and interpretation of results were our own.
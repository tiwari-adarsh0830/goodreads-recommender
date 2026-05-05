# Capstone Walkthrough — DSGA1004 Big Data, Spring 2026
**NetID:** at7095 

This is my personal walkthrough of everything I did, in the order I did it, with the actual results. The README has the formal project description — this is the real story of what happened.

---

## The Data

The Goodreads dataset has ~223M user-book interactions across ~876K users and ~2.4M books. Each row is `user_id, book_id, is_read, rating, is_reviewed`. The three fields mean very different things:

- `rating` — explicit, the user actually scored the book 1–5. Only 45.7% of interactions have this.
- `is_read` — implicit, the user marked it as read. Doesn't mean they liked it.
- `is_reviewed` — implicit engagement, they wrote something. Also doesn't mean they liked it.

This distinction ended up being the most important thing in the whole project. More on that below.

First thing I did was convert the raw CSV to Parquet. The CSV is huge and slow to repeatedly scan — Parquet made every downstream job significantly faster.

---

## Data Split (step1)

Split was per-user, 70/15/15 train/val/test using a seeded random ordering of each user's interactions. Users with fewer than 10 interactions were kept in train only — you can't meaningfully evaluate recommendations for someone with almost no history.

- Total eligible users: 772,730
- Ineligible (train only): 103,415
- seed=1004 throughout for reproducibility

For prototyping I also created a small split — 1% sample of users (~9,026 users), same proportions. All the ALS eval runs on this small split because the full dataset kept hitting the 6-hour YARN cluster limit.

---

## D1 — Market Segmentation with MinHash/LSH (step2)

The goal was to find the 100 most similar book pairs by user overlap — books read by the same groups of people.

I used MinHash with LSH (Locality Sensitive Hashing) to approximate Jaccard similarity at scale. Computing exact Jaccard over 2.4M books is completely infeasible — MinHash lets you do it in linear time with a tunable accuracy tradeoff.

**Parameters:**
- `MIN_READERS = 200` — filtered out books with fewer than 200 readers. Books with tiny readerships produce noisy similarity estimates and aren't useful for market segmentation.
- `NUM_HASH_TABLES = 32`
- `LSH_THRESHOLD = 0.3`
- Books surviving filter: 67,100

**Results:** Top-100 pairs have Jaccard similarity ranging from 0.8963 to 0.9502. Full list in `step2_top100_pairs.csv`, sorted descending.

**Interesting finding:** The top pairs are almost entirely adjacent book IDs — e.g. book 1234 and book 1235. These are different editions of the same work (hardcover vs paperback, different publishers, etc.). Goodreads treats each edition as a separate book ID, so the same readers show up on both. This is a property of how Goodreads structures its data, not a bug in the algorithm. The MinHash is working correctly — these really are the most similar pairs by user overlap.

---

## D2 — Popularity Baseline (step3)

Before building anything complicated, I needed a simple baseline to beat.

I tried two variants:

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

---

## D4 — Implicit ALS (step5) — Main Model

This is where things actually worked.

Switched to `implicitPrefs=True`. Instead of predicting a rating, the model learns from binary consumption signals using a confidence-weighted matrix factorization.

**Signal:** `confidence = 1 + alpha * is_read`  
**alpha = 40.0** — this means a book the user read gets 41x the confidence weight of an unread interaction. Tried a few values; 40 gave the best val performance.

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

## D5 — Combined Explicit + Implicit (step6) — Best Model

The idea: when a user has both read AND rated a book, the rating tells you how much they liked it. Pure `is_read` treats a 1-star read the same as a 5-star read. That's information being thrown away.

**Signal:**
```
if rating > 0:
    combined = (rating / 5.0) * is_read + 0.3 * is_reviewed
else:
    combined = 0.5 * is_read + 0.3 * is_reviewed
```

When rated: `rating/5.0` maps to [0.2, 1.0] — encodes preference strength. A 5-star read gets full weight, a 1-star read gets 0.2. When unrated: 0.5 is a neutral prior, assuming the read was neither strongly positive nor negative. `is_reviewed` gets 0.3 weight throughout — small enough to add signal without dominating.

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

## What I'd Do Differently

- Run full val/test eval for D4 and D5 on all 772K users if given more cluster time
- Try higher alpha values (60, 80) for implicit ALS
- Experiment with different weights for `is_reviewed` in D5
- Try filtering `is_read=1` only for training instead of all interactions

---

## AI Disclosure

Claude (Anthropic) was used extensively throughout this project — for writing and debugging PySpark scripts, diagnosing the D3 eval bug, and structuring the report. All experimental design decisions, parameter choices, and result interpretation are my own.

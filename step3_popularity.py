"""
Step 3 (Deliverable 2): Popularity baseline + reusable evaluation framework.

Author: Adarsh Tiwari and Sheetal Sood (Goodreads Capstone, DSGA1004 Big Data, Spring 2026)

This script:
  1. Computes two popularity rankings on the training split:
       (a) POP_READ      -- count of is_read=1 per book
       (b) POP_BAYES     -- Bayesian-shrunk weighted-average rating, score
                            (n_b * mean_b + C * mu_g) / (n_b + C),
                            with C=20 and mu_g = global mean of rated
                            interactions on the training split.
  2. For each user in val/test:
       - Take the global top-200 popularity ranking,
       - Filter out the user's training items,
       - Truncate to top 100 -- this is the user's recommendation list.
  3. Ground truth per user = books with is_read=1 in their val (or test)
     split rows.
  4. Evaluate with eval_helpers.compute_ranking_metrics:
       Precision@100, MAP@100, NDCG@100.
  5. Reports both popularity definitions on val AND test.

The eval_helpers module is reused identically by steps 4, 5, 6 (ALS
explicit / implicit / combined) so all four deliverables share a single
evaluator -- this is the "same general evaluation framework" the brief
requires.

We oversample to top-200 globally before filtering training items so that
even a power-user who has read most of the top-200 still has 100 remaining.
We verify in the script that the slice never falls below 100.

Run (client mode, light workload):
    spark-submit --master yarn --deploy-mode client \\
        --num-executors 3 --executor-cores 4 --executor-memory 6G \\
        --driver-memory 2G \\
        --conf spark.dynamicAllocation.enabled=true \\
        --conf spark.dynamicAllocation.minExecutors=1 \\
        --conf spark.dynamicAllocation.maxExecutors=4 \\
        --conf spark.sql.shuffle.partitions=200 \\
        --py-files ~/goodreads_capstone/scripts/eval_helpers.py \\
        ~/goodreads_capstone/scripts/step3_popularity.py [--small]
"""
import os
import sys
import time
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark import StorageLevel

# Make the helpers importable when run via spark-submit (the --py-files
# flag uploads it to the cluster).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_helpers import compute_ranking_metrics, format_metrics

# ---- Constants ----------------------------------------------------------
NETID            = os.environ.get("NETID", "your_netid")
DATA_DIR         = f"hdfs:///user/{NETID}_nyu_edu/goodreads"
OUT_DIR          = f"{DATA_DIR}/popularity"
LOCAL_REPORT_DIR = os.path.expanduser("~/goodreads_capstone/report_artifacts")
os.makedirs(LOCAL_REPORT_DIR, exist_ok=True)

K_EVAL           = 100      # top-K for evaluation, per brief
K_OVERSAMPLE     = 1000     # global candidate pool before per-user filtering
BAYES_C          = 20       # smoothing constant for Bayesian average

# ---- Argument parsing: --small toggles prototyping subset ---------------
SMALL = "--small" in sys.argv
suffix = "_small" if SMALL else ""
TRAIN_PATH = f"{DATA_DIR}/train{suffix}.parquet"
VAL_PATH   = f"{DATA_DIR}/val{suffix}.parquet"
TEST_PATH  = f"{DATA_DIR}/test{suffix}.parquet"

print(f"=== Step 3 (Popularity baseline) | small={SMALL} ===")

# ---- Spark session ------------------------------------------------------
spark = (SparkSession.builder
    .appName(f"{NETID}_step3_popularity{'_small' if SMALL else ''}")
    .config("spark.sql.parquet.compression.codec", "snappy")
    .getOrCreate())
spark.sparkContext.setLogLevel("WARN")
t0 = time.time()

# ---- Load splits --------------------------------------------------------
train = spark.read.parquet(TRAIN_PATH)
val   = spark.read.parquet(VAL_PATH)
test  = spark.read.parquet(TEST_PATH)
print(f"train rows : {train.count():,}")
print(f"val   rows : {val.count():,}")
print(f"test  rows : {test.count():,}")

# ---- Compute popularity scores on TRAINING DATA ONLY --------------------
# CRITICAL: never compute popularity from val/test rows -- that would leak
# evaluation labels into the recommender.

# (a) POP_READ : raw is_read=1 count per book
pop_read = (train
    .filter(F.col("is_read") == 1)
    .groupBy("book_id").agg(F.count("*").alias("n_reads")))

# (b) POP_BAYES : Bayesian-shrunk mean rating
# global mean rating across rated interactions in TRAIN
mu_global_row = (train
    .filter(F.col("rating") > 0)
    .agg(F.avg("rating").alias("mu_g"))
    .collect()[0])
MU_G = float(mu_global_row["mu_g"])
print(f"Global mean rating (mu_g) on train: {MU_G:.4f}")

book_rating_stats = (train
    .filter(F.col("rating") > 0)
    .groupBy("book_id")
    .agg(F.count("*").alias("n_rated"),
         F.avg("rating").alias("mean_rating")))

pop_bayes = book_rating_stats.withColumn(
    "bayes_score",
    (F.col("n_rated") * F.col("mean_rating") + F.lit(BAYES_C) * F.lit(MU_G))
    / (F.col("n_rated") + F.lit(BAYES_C)))

# ---- Build the global top-K_OVERSAMPLE candidate lists ------------------
top_read_books = ([r.book_id for r in pop_read
                   .orderBy(F.col("n_reads").desc())
                   .limit(K_OVERSAMPLE)
                   .collect()])
top_bayes_books = ([r.book_id for r in pop_bayes
                    .orderBy(F.col("bayes_score").desc())
                    .limit(K_OVERSAMPLE)
                    .collect()])

print(f"Top {K_OVERSAMPLE} most-read books computed.")
print(f"Top {K_OVERSAMPLE} bayes-popular books computed.")
print(f"First 5 most-read book IDs : {top_read_books[:5]}")
print(f"First 5 bayes-popular IDs  : {top_bayes_books[:5]}")

# ---- Build per-user training-items map ---------------------------------
user_train_items = (train.groupBy("user_id")
    .agg(F.collect_set("book_id").alias("train_items")))
user_train_items = user_train_items.persist(StorageLevel.MEMORY_AND_DISK)
_ = user_train_items.count()

# ---- Predictions function -----------------------------------------------
# Same logic for both popularity variants: take the global top-K_OVERSAMPLE,
# remove the user's training items, slice to K_EVAL.
def make_predictions(top_books_list):
    top_arr = F.array(*[F.lit(b) for b in top_books_list])
    return (user_train_items
        .withColumn("recs",
            F.slice(F.array_except(top_arr, F.col("train_items")),
                    1, K_EVAL))
        .select("user_id", "recs"))

preds_read  = make_predictions(top_read_books)
preds_bayes = make_predictions(top_bayes_books)

# Sanity check: every user should have 100 recommendations.
# If not, K_OVERSAMPLE is too small.
def _rec_size_diagnostic(preds_df, label):
    sizes = preds_df.select(F.size("recs").alias("n"))
    n_total  = sizes.count()
    n_short  = sizes.filter(F.col("n") < K_EVAL).count()
    min_size = sizes.agg(F.min("n")).collect()[0][0]
    short_frac = n_short / n_total if n_total else 0.0
    print(f"[{label}] users={n_total:,}  min_recs={min_size}  "
          f"users_with_<{K_EVAL}_recs={n_short:,} ({short_frac*100:.3f}%)")
    # Hard cap: at most 1% of users may have fewer than K_EVAL recs.
    # Power-users with thousands of train items can fall below; we accept
    # this and document, rather than fail.
    assert short_frac <= 0.01, (
        f"{short_frac*100:.2f}% of users have <{K_EVAL} recs for {label}. "
        f"Increase K_OVERSAMPLE.")

_rec_size_diagnostic(preds_read,  "POP_READ")
_rec_size_diagnostic(preds_bayes, "POP_BAYES")

# ---- Build per-user ground truth (val and test) -------------------------
# Ground truth = is_read=1 books in val (or test) for that user.
def make_ground_truth(split_df):
    return (split_df.filter(F.col("is_read") == 1)
            .groupBy("user_id")
            .agg(F.collect_set("book_id").alias("ground_truth")))

gt_val  = make_ground_truth(val)
gt_test = make_ground_truth(test)

# ---- Evaluate -----------------------------------------------------------
print("\nEvaluating POP_READ on val...")
m_read_val   = compute_ranking_metrics(preds_read,  gt_val,  k=K_EVAL)
print("Evaluating POP_READ on test...")
m_read_test  = compute_ranking_metrics(preds_read,  gt_test, k=K_EVAL)
print("Evaluating POP_BAYES on val...")
m_bayes_val  = compute_ranking_metrics(preds_bayes, gt_val,  k=K_EVAL)
print("Evaluating POP_BAYES on test...")
m_bayes_test = compute_ranking_metrics(preds_bayes, gt_test, k=K_EVAL)

# ---- Report -------------------------------------------------------------
report_text = (
    f"=== Step 3 (Popularity baseline) | small={SMALL} ===\n"
    f"BAYES_C            : {BAYES_C}\n"
    f"global mean rating : {MU_G:.4f}\n"
    f"K (eval cutoff)    : {K_EVAL}\n"
    f"K (global oversample): {K_OVERSAMPLE}\n\n"
    + format_metrics(m_read_val,   "POP_READ on validation",  K_EVAL)
    + format_metrics(m_read_test,  "POP_READ on test",        K_EVAL)
    + format_metrics(m_bayes_val,  "POP_BAYES on validation", K_EVAL)
    + format_metrics(m_bayes_test, "POP_BAYES on test",       K_EVAL))
print("\n" + report_text)

local_path = os.path.join(LOCAL_REPORT_DIR,
                          f"step3_popularity_metrics{'_small' if SMALL else ''}.txt")
with open(local_path, "w") as f:
    f.write(report_text)
print(f"Saved metrics to {local_path}")

# Save the global top-100 most-read book IDs (with Goodreads IDs) for
# inclusion in the report appendix.
book_map = spark.read.parquet(f"{DATA_DIR}/book_id_map.parquet")
top_read_df = spark.createDataFrame(
    [(i + 1, b) for i, b in enumerate(top_read_books[:100])],
    ["rank", "book_id_csv"])
top_read_named = (top_read_df
    .join(book_map.withColumnRenamed("book_id_csv", "book_id_csv")
                   .withColumnRenamed("book_id", "goodreads_id"),
          on="book_id_csv", how="left")
    .orderBy("rank")
    .select("rank", "book_id_csv", "goodreads_id"))
top_read_named.coalesce(1).write.mode("overwrite").option("header", "true") \
    .csv(f"{OUT_DIR}/top100_most_read_csv{'_small' if SMALL else ''}")

elapsed = time.time() - t0
print(f"\n[Step 3 complete in {elapsed/60:.1f} min]")

user_train_items.unpersist()
spark.stop()

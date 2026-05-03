"""
step4_als_explicit_tune.py
--------------------------
Deliverable 3 — Explicit-feedback ALS: Hyperparameter tuning on small split.

Grid search: rank ∈ {10, 20, 50, 100} × regParam ∈ {0.01, 0.1, 1.0} = 12 combos
Tuning metric: MAP@100 on small val set (is_read=1 ground truth)

eval_helpers interface:
  predictions  : user_id (int), recs (array<int>)  — ranked best-first
  ground_truth : user_id (int), ground_truth (array<int>)

ORDERING FIX: collect_list after orderBy is not guaranteed in distributed Spark.
We collect structs (pred_score, book_id), sort the array explicitly with
array_sort (descending pred_score), then extract book_ids. This guarantees
the recs array is correctly ranked best-first regardless of partition order.

Run:
  spark-submit --master yarn --deploy-mode client \
    --num-executors 3 --executor-cores 4 --executor-memory 6G --driver-memory 2G \
    --conf spark.dynamicAllocation.enabled=true \
    --conf spark.dynamicAllocation.maxExecutors=4 \
    --conf spark.sql.shuffle.partitions=200 \
    --conf spark.sql.autoBroadcastJoinThreshold=-1 \
    --py-files ~/goodreads_capstone/scripts/eval_helpers.py \
    ~/goodreads_capstone/scripts/step4_als_explicit_tune.py \
    2>&1 | tee ~/goodreads_capstone/logs/step4_tune_$(date +%Y%m%d_%H%M%S).log
"""

import sys, os
from itertools import product

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_helpers import compute_ranking_metrics

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE    = "hdfs:///user/at7095_nyu_edu/goodreads"
TRAIN_S = f"{BASE}/train_small.parquet"
VAL_S   = f"{BASE}/val_small.parquet"
OUT_DIR = f"{BASE}/als_explicit/tune"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANKS      = [10, 20, 50, 100]
REG_PARAMS = [0.01, 0.1, 1.0]
MAX_ITER   = 10
SEED       = 1004
K_OVER     = 1000
K_EVAL     = 100

# ---------------------------------------------------------------------------
# Spark
# ---------------------------------------------------------------------------
spark = (SparkSession.builder
         .appName("D3_ALS_Explicit_Tune")
         .config("spark.sql.autoBroadcastJoinThreshold", "-1")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

# ---------------------------------------------------------------------------
# Load small splits
# ---------------------------------------------------------------------------
train_raw = spark.read.parquet(TRAIN_S)
val_raw   = spark.read.parquet(VAL_S)

# ---------------------------------------------------------------------------
# ALS training data: explicit feedback only (rating > 0)
# ---------------------------------------------------------------------------
train_explicit = (
    train_raw
    .filter(F.col("rating") > 0)
    .select(
        F.col("user_id"),
        F.col("book_id"),
        F.col("rating").cast("float"),
    )
    .cache()
)
print(f"[tune] Explicit training rows (rating>0): {train_explicit.count():,}")

# All training interactions for seen-item exclusion
train_seen = (
    train_raw
    .select("user_id", "book_id")
    .distinct()
    .cache()
)

# ---------------------------------------------------------------------------
# Val ground truth: user_id, ground_truth (array<int>)
# ---------------------------------------------------------------------------
val_gt = (
    val_raw
    .filter(F.col("is_read") == 1)
    .groupBy("user_id")
    .agg(F.collect_set("book_id").alias("ground_truth"))
    .cache()
)
print(f"[tune] Val ground-truth users: {val_gt.count():,}")

val_users = val_gt.select("user_id").cache()

# ---------------------------------------------------------------------------
# Helper: exploded recs → correctly ordered array per user
#
# WHY NOT orderBy then collect_list:
#   Spark's groupBy shuffle does not preserve row order. collect_list picks
#   up rows in whatever order they arrive at the reducer — arbitrary.
#
# CORRECT APPROACH:
#   1. Collect structs {pred_score, book_id} (order doesn't matter here)
#   2. array_sort sorts descending by pred_score (negate for desc)
#   3. transform extracts book_id from each struct → final ranked array
# ---------------------------------------------------------------------------
def make_predictions(recs_exp, train_seen, k):
    """
    recs_exp : user_id, book_id, pred_score  (one row per rec, unordered)
    Returns  : user_id, recs (array<int>, best-first, length <= k)
    """
    # Exclude training interactions
    recs_new = recs_exp.join(train_seen, on=["user_id", "book_id"], how="left_anti")

    # Collect (neg_score, book_id) structs per user; array_sort sorts ascending
    # by first field, so negating pred_score gives descending order.
    predictions = (
        recs_new
        .select(
            "user_id",
            F.struct(
                (-F.col("pred_score")).alias("neg_score"),
                F.col("book_id")
            ).alias("rec_struct")
        )
        .groupBy("user_id")
        .agg(F.collect_list("rec_struct").alias("recs_raw"))
        .withColumn("recs_sorted", F.array_sort("recs_raw"))   # sort by neg_score asc = score desc
        .withColumn("recs_cut", F.slice("recs_sorted", 1, k))  # top-k
        .withColumn("recs", F.expr("transform(recs_cut, x -> x.book_id)"))
        .select("user_id", "recs")
    )
    return predictions

# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------
results = []

for rank, reg in product(RANKS, REG_PARAMS):
    print(f"\n{'='*56}")
    print(f"[tune] rank={rank}  regParam={reg}")

    als = ALS(
        rank=rank,
        maxIter=MAX_ITER,
        regParam=reg,
        userCol="user_id",
        itemCol="book_id",
        ratingCol="rating",
        implicitPrefs=False,
        coldStartStrategy="drop",
        seed=SEED,
        nonnegative=False,
    )
    model = als.fit(train_explicit)

    # Warm val users only
    model_users = model.userFactors.select(F.col("id").alias("user_id"))
    warm_val = val_users.join(model_users, on="user_id", how="inner")
    n_warm = warm_val.count()
    print(f"[tune]   Warm val users: {n_warm:,}")

    if n_warm == 0:
        print("[tune]   No warm val users — recording zeros.")
        results.append({"rank": rank, "regParam": reg,
                        "precision_100": 0.0, "map_100": 0.0,
                        "ndcg_100": 0.0, "n_users": 0})
        del model
        continue

    # Top-K_OVER raw recs: user_id, recommendations array<struct<book_id,rating>>
    raw_recs = model.recommendForUserSubset(warm_val, K_OVER)

    # Explode to flat rows
    recs_exp = (
        raw_recs
        .select("user_id", F.explode("recommendations").alias("rec"))
        .select(
            "user_id",
            F.col("rec.book_id").alias("book_id"),
            F.col("rec.rating").alias("pred_score"),
        )
    )

    # Build correctly ordered predictions array
    predictions = make_predictions(recs_exp, train_seen, K_EVAL)

    metrics = compute_ranking_metrics(predictions, val_gt, k=K_EVAL)
    row = {
        "rank":          rank,
        "regParam":      reg,
        "precision_100": round(metrics["precision_at_k"], 6),
        "map_100":       round(metrics["map_at_k"],       6),
        "ndcg_100":      round(metrics["ndcg_at_k"],      6),
        "n_users":       int(metrics["n_users_eval"]),
    }
    results.append(row)
    print(f"[tune]   P@100={row['precision_100']:.4f}  "
          f"MAP@100={row['map_100']:.4f}  "
          f"NDCG@100={row['ndcg_100']:.4f}  "
          f"n_users={row['n_users']}")
    del model

# ---------------------------------------------------------------------------
# Write results to HDFS
# ---------------------------------------------------------------------------
header   = ["rank", "regParam", "precision_100", "map_100", "ndcg_100", "n_users"]
rows_str = [",".join(header)] + [",".join(str(r[h]) for h in header) for r in results]

(spark.createDataFrame([(s,) for s in rows_str], ["value"])
      .coalesce(1)
      .write.mode("overwrite")
      .text(f"{OUT_DIR}/tune_results.csv"))

print(f"\n[tune] Results → {OUT_DIR}/tune_results.csv")

# ---------------------------------------------------------------------------
# Print sorted grid and best config
# ---------------------------------------------------------------------------
print(f"\n{'rank':>6} {'regParam':>10} {'P@100':>8} {'MAP@100':>9} {'NDCG@100':>10} {'n_users':>9}")
for r in sorted(results, key=lambda x: x["map_100"], reverse=True):
    print(f"{r['rank']:>6} {r['regParam']:>10} "
          f"{r['precision_100']:>8.4f} {r['map_100']:>9.4f} "
          f"{r['ndcg_100']:>10.4f} {r['n_users']:>9}")

best = max(results, key=lambda x: x["map_100"])
print(f"\n[tune] BEST → rank={best['rank']}, regParam={best['regParam']}, "
      f"MAP@100={best['map_100']:.6f}")

(spark.createDataFrame(
    [(f"rank={best['rank']}",), (f"regParam={best['regParam']}",)],
    ["value"])
 .coalesce(1)
 .write.mode("overwrite")
 .text(f"{OUT_DIR}/best_params.txt"))

print(f"[tune] Best params → {OUT_DIR}/best_params.txt")
spark.stop()
print("[tune] Done.")

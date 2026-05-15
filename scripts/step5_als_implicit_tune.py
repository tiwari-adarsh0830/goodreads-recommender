"""
step5_als_implicit_tune.py
--------------------------
Deliverable 4 follow-up: Hyperparameter sweep for implicit-feedback ALS.

D4 in the report reused the (rank, regParam) winner from the D3 explicit
grid and only spot-checked alpha. This script does a proper small sweep
on the implicit objective:

  (rank, regParam) grid  : rank in {20, 50, 100}, regParam in {0.01, 0.1, 1.0}
                           fixed alpha = 40, fixed maxIter = 10
  alpha sweep at best    : alpha in {10, 20, 40, 60} at the best (rank, reg)

Tuning metric: MAP@100 on the small val split, is_read=1 ground truth.

The point is to test whether the implicit-feedback loss prefers a
different (rank, regParam) than the explicit-feedback loss did, and to
confirm that alpha=40 is on the flat part of the curve rather than the
edge.

Run:
  spark-submit --master yarn --deploy-mode client \\
    --num-executors 3 --executor-cores 4 --executor-memory 6G --driver-memory 2G \\
    --conf spark.dynamicAllocation.enabled=true \\
    --conf spark.dynamicAllocation.maxExecutors=4 \\
    --conf spark.sql.shuffle.partitions=200 \\
    --conf spark.sql.autoBroadcastJoinThreshold=-1 \\
    --py-files ~/goodreads_capstone/scripts/eval_helpers.py \\
    ~/goodreads_capstone/scripts/step5_als_implicit_tune.py \\
    2>&1 | tee ~/goodreads_capstone/logs/step5_tune_$(date +%Y%m%d_%H%M%S).log
"""

import sys, os
from itertools import product

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_helpers import compute_ranking_metrics

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
NETID   = os.environ.get("NETID", "ss21040")
BASE    = f"hdfs:///user/{NETID}_nyu_edu/goodreads"
TRAIN_S = f"{BASE}/train_small.parquet"
VAL_S   = f"{BASE}/val_small.parquet"
OUT_DIR = f"{BASE}/als_implicit/tune"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANKS      = [20, 50, 100]
REG_PARAMS = [0.01, 0.1, 1.0]
ALPHAS     = [10, 20, 40, 60]
ALPHA_FIX  = 40.0
MAX_ITER   = 10
SEED       = 1004
K_OVER     = 1000
K_EVAL     = 100

# ---------------------------------------------------------------------------
# Spark
# ---------------------------------------------------------------------------
spark = (SparkSession.builder
         .appName("D4_ALS_Implicit_Tune")
         .config("spark.sql.autoBroadcastJoinThreshold", "-1")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

# ---------------------------------------------------------------------------
# Load small splits
# ---------------------------------------------------------------------------
train_raw = spark.read.parquet(TRAIN_S)
val_raw   = spark.read.parquet(VAL_S)

# Implicit training data: ALL interactions, is_read as the binary signal.
# Same construction as step5_als_implicit_train.py so the sweep is
# directly comparable to the headline D4 model.
train_implicit = (
    train_raw
    .select(
        F.col("user_id"),
        F.col("book_id"),
        F.col("is_read").cast("float").alias("is_read"),
    )
    .cache()
)
print(f"[tune] Implicit training rows : {train_implicit.count():,}")

train_seen = (
    train_raw
    .select("user_id", "book_id")
    .distinct()
    .cache()
)

val_gt = (
    val_raw
    .filter(F.col("is_read") == 1)
    .groupBy("user_id")
    .agg(F.collect_set("book_id").alias("ground_truth"))
    .cache()
)
print(f"[tune] Val ground-truth users : {val_gt.count():,}")

val_users = val_gt.select("user_id").cache()


def make_predictions(recs_exp, train_seen, k):
    """
    Same ordered top-K construction as step4_als_explicit_tune.py.
    Avoids the orderBy + collect_list ordering pitfall by sorting an
    array of structs explicitly.
    """
    recs_new = recs_exp.join(train_seen, on=["user_id", "book_id"], how="left_anti")
    return (recs_new
        .select(
            "user_id",
            F.struct(
                (-F.col("pred_score")).alias("neg_score"),
                F.col("book_id"),
            ).alias("rec_struct"),
        )
        .groupBy("user_id")
        .agg(F.collect_list("rec_struct").alias("recs_raw"))
        .withColumn("recs_sorted", F.array_sort("recs_raw"))
        .withColumn("recs_cut", F.slice("recs_sorted", 1, k))
        .withColumn("recs", F.expr("transform(recs_cut, x -> x.book_id)"))
        .select("user_id", "recs"))


def fit_and_score(rank, reg, alpha):
    als = ALS(
        rank=rank,
        maxIter=MAX_ITER,
        regParam=reg,
        alpha=alpha,
        implicitPrefs=True,
        userCol="user_id",
        itemCol="book_id",
        ratingCol="is_read",
        coldStartStrategy="drop",
        seed=SEED,
        nonnegative=False,
    )
    model = als.fit(train_implicit)

    model_users = model.userFactors.select(F.col("id").alias("user_id"))
    warm_val = val_users.join(model_users, on="user_id", how="inner")
    n_warm = warm_val.count()
    if n_warm == 0:
        del model
        return {"precision_100": 0.0, "map_100": 0.0,
                "ndcg_100": 0.0, "n_users": 0}

    raw_recs = model.recommendForUserSubset(warm_val, K_OVER)
    recs_exp = (
        raw_recs
        .select("user_id", F.explode("recommendations").alias("rec"))
        .select(
            "user_id",
            F.col("rec.book_id").alias("book_id"),
            F.col("rec.rating").alias("pred_score"),
        )
    )
    preds = make_predictions(recs_exp, train_seen, K_EVAL)
    m = compute_ranking_metrics(preds, val_gt, k=K_EVAL)
    out = {
        "precision_100": round(m["precision_at_k"], 6),
        "map_100":       round(m["map_at_k"], 6),
        "ndcg_100":      round(m["ndcg_at_k"], 6),
        "n_users":       int(m["n_users_eval"]),
    }
    del model
    return out


# ---------------------------------------------------------------------------
# Phase A: (rank, regParam) grid at fixed alpha
# ---------------------------------------------------------------------------
grid_results = []
for rank, reg in product(RANKS, REG_PARAMS):
    print(f"\n{'='*56}\n[tune] PHASE-A rank={rank}  regParam={reg}  alpha={ALPHA_FIX}")
    out = fit_and_score(rank, reg, ALPHA_FIX)
    out.update({"rank": rank, "regParam": reg, "alpha": ALPHA_FIX, "phase": "A"})
    grid_results.append(out)
    print(f"[tune]   P@100={out['precision_100']:.4f}  "
          f"MAP@100={out['map_100']:.4f}  "
          f"NDCG@100={out['ndcg_100']:.4f}  n_users={out['n_users']}")

best_grid = max(grid_results, key=lambda r: r["map_100"])
print(f"\n[tune] PHASE-A best: rank={best_grid['rank']}, "
      f"regParam={best_grid['regParam']}, MAP@100={best_grid['map_100']:.6f}")

# ---------------------------------------------------------------------------
# Phase B: alpha sweep at the winning (rank, regParam)
# ---------------------------------------------------------------------------
alpha_results = []
for a in ALPHAS:
    print(f"\n{'='*56}\n[tune] PHASE-B alpha={a} at rank={best_grid['rank']} regParam={best_grid['regParam']}")
    out = fit_and_score(best_grid["rank"], best_grid["regParam"], float(a))
    out.update({"rank": best_grid["rank"], "regParam": best_grid["regParam"],
                "alpha": float(a), "phase": "B"})
    alpha_results.append(out)
    print(f"[tune]   P@100={out['precision_100']:.4f}  "
          f"MAP@100={out['map_100']:.4f}  "
          f"NDCG@100={out['ndcg_100']:.4f}  n_users={out['n_users']}")

# ---------------------------------------------------------------------------
# Write results
# ---------------------------------------------------------------------------
all_results = grid_results + alpha_results
header   = ["phase", "rank", "regParam", "alpha", "precision_100",
            "map_100", "ndcg_100", "n_users"]
rows_str = [",".join(header)] + [",".join(str(r[h]) for h in header) for r in all_results]

(spark.createDataFrame([(s,) for s in rows_str], ["value"])
      .coalesce(1)
      .write.mode("overwrite")
      .text(f"{OUT_DIR}/tune_results.csv"))

print(f"\n[tune] Results written -> {OUT_DIR}/tune_results.csv")

# Sorted print
print(f"\n{'phase':>6} {'rank':>5} {'regParam':>9} {'alpha':>6} "
      f"{'P@100':>8} {'MAP@100':>9} {'NDCG@100':>10} {'n_users':>9}")
for r in sorted(all_results, key=lambda x: x["map_100"], reverse=True):
    print(f"{r['phase']:>6} {r['rank']:>5} {r['regParam']:>9} {r['alpha']:>6} "
          f"{r['precision_100']:>8.4f} {r['map_100']:>9.4f} "
          f"{r['ndcg_100']:>10.4f} {r['n_users']:>9}")

best_overall = max(all_results, key=lambda x: x["map_100"])
print(f"\n[tune] BEST OVERALL -> rank={best_overall['rank']}, "
      f"regParam={best_overall['regParam']}, alpha={best_overall['alpha']}, "
      f"MAP@100={best_overall['map_100']:.6f}")

(spark.createDataFrame(
    [(f"rank={best_overall['rank']}",),
     (f"regParam={best_overall['regParam']}",),
     (f"alpha={best_overall['alpha']}",),
     (f"val_map_100={best_overall['map_100']:.6f}",)],
    ["value"])
 .coalesce(1)
 .write.mode("overwrite")
 .text(f"{OUT_DIR}/best_params.txt"))

print(f"[tune] Best params -> {OUT_DIR}/best_params.txt")
spark.stop()
print("[tune] Done.")

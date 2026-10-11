"""
step4_als_explicit_full.py
--------------------------
Deliverable 3 — Explicit-feedback ALS: Full training + evaluation.

Memory-safe version: recommendations generated in user batches to avoid
OOM during recommendForUserSubset on 742K users. Results written to HDFS
as parquet, then read back for metric computation.

Run (cluster mode):
  spark-submit --master yarn --deploy-mode cluster \
    --num-executors 3 --executor-cores 4 --executor-memory 6G \
    --driver-cores 2 --driver-memory 6G \
    --conf spark.dynamicAllocation.enabled=true \
    --conf spark.dynamicAllocation.maxExecutors=4 \
    --conf spark.sql.shuffle.partitions=600 \
    --conf spark.sql.autoBroadcastJoinThreshold=-1 \
    --conf spark.driver.maxResultSize=0 \
    --conf spark.network.timeout=600s \
    --conf spark.executor.heartbeatInterval=60s \
    --conf spark.yarn.maxAppAttempts=1 \
    --py-files ~/goodreads_capstone/scripts/eval_helpers.py \
    ~/goodreads_capstone/scripts/step4_als_explicit_full.py
"""

import sys, os

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_helpers import compute_ranking_metrics

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE     = f"hdfs:///user/{os.environ.get('NETID', 'your_netid')}_nyu_edu/goodreads"
TRAIN    = f"{BASE}/train.parquet"
VAL      = f"{BASE}/val.parquet"
TEST     = f"{BASE}/test.parquet"
TUNE_DIR = f"{BASE}/als_explicit/tune"
OUT_DIR  = f"{BASE}/als_explicit/full"

# ---------------------------------------------------------------------------
# Hyperparameters — hard-coded after tuning
# ---------------------------------------------------------------------------
BEST_RANK = 50
BEST_REG  = 0.1
MAX_ITER  = 10
SEED      = 1004
K_OVER    = 500    # reduced from 1000 to save memory; still 5x K_EVAL
K_EVAL    = 100
N_PARTS   = 200    # repartition users into this many partitions for recs

# ---------------------------------------------------------------------------
# Spark
# ---------------------------------------------------------------------------
spark = (SparkSession.builder
         .appName("D3_ALS_Explicit_Full")
         .config("spark.sql.autoBroadcastJoinThreshold", "-1")
         .config("spark.driver.maxResultSize", "0")
         .config("spark.sql.shuffle.partitions", "600")
         .getOrCreate())
spark.sparkContext.setCheckpointDir(f"hdfs:///user/{os.environ.get('NETID', 'your_netid')}_nyu_edu/goodreads/checkpoints")
spark.sparkContext.setLogLevel("WARN")

print(f"[full] Hyperparameters: rank={BEST_RANK}, regParam={BEST_REG}")

# ---------------------------------------------------------------------------
# Load full splits
# ---------------------------------------------------------------------------
print("[full] Loading full splits...")
train_raw = spark.read.parquet(TRAIN)
val_raw   = spark.read.parquet(VAL)
test_raw  = spark.read.parquet(TEST)

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
    .repartition(N_PARTS, "user_id")
    .cache()
)
n_expl = train_explicit.count()
print(f"[full] Explicit training rows: {n_expl:,}")

# All training interactions for seen-item exclusion
# Write to HDFS and read back to avoid keeping huge DF in memory
train_seen_path = f"{OUT_DIR}/train_seen_tmp"
(train_raw
 .select("user_id", "book_id")
 .distinct()
 .write.mode("overwrite").parquet(train_seen_path))
train_seen = spark.read.parquet(train_seen_path)

# ---------------------------------------------------------------------------
# Ground truth: user_id, ground_truth (array<int>)
# Write to HDFS to avoid recomputation and memory pressure
# ---------------------------------------------------------------------------
def make_and_save_gt(df, path):
    (df
     .filter(F.col("is_read") == 1)
     .groupBy("user_id")
     .agg(F.collect_set("book_id").alias("ground_truth"))
     .write.mode("overwrite").parquet(path))

val_gt_path  = f"{OUT_DIR}/val_gt_tmp"
test_gt_path = f"{OUT_DIR}/test_gt_tmp"
make_and_save_gt(val_raw,  val_gt_path)
make_and_save_gt(test_raw, test_gt_path)

val_gt  = spark.read.parquet(val_gt_path)
test_gt = spark.read.parquet(test_gt_path)
print(f"[full] Val  ground-truth users: {val_gt.count():,}")
print(f"[full] Test ground-truth users: {test_gt.count():,}")

# ---------------------------------------------------------------------------
# Train ALS
# ---------------------------------------------------------------------------
print(f"\n[full] Training ALS: rank={BEST_RANK}, regParam={BEST_REG}, maxIter={MAX_ITER}")
als = ALS(
    rank=BEST_RANK,
    maxIter=MAX_ITER,
    regParam=BEST_REG,
    userCol="user_id",
    itemCol="book_id",
    ratingCol="rating",
    implicitPrefs=False,
    coldStartStrategy="drop",
    seed=SEED,
    nonnegative=False,
)
model = als.fit(train_explicit)
print("[full] Training complete.")

# Free training data from memory — no longer needed
train_explicit.unpersist()

# Save model
model.write().overwrite().save(f"{OUT_DIR}/model")
print(f"[full] Model saved → {OUT_DIR}/model")

# ---------------------------------------------------------------------------
# Generate recommendations for a split and save to HDFS as predictions parquet
# Returns path to saved predictions parquet
# ---------------------------------------------------------------------------
model_users = model.userFactors.select(F.col("id").alias("user_id"))

def generate_and_save_predictions(gt_df, split_name, out_path):
    """
    Generates top-K_EVAL recommendations per user, saves to HDFS.
    Uses recommendForUserSubset on all warm users at once, but writes
    results incrementally to avoid driver OOM.
    """
    warm = (gt_df.select("user_id")
                 .join(model_users, on="user_id", how="inner")
                 .repartition(N_PARTS))
    n_warm = warm.count()
    print(f"[full] {split_name}: warm users = {n_warm:,}")

    # Generate raw recs — Spark handles this lazily partition by partition
    raw_recs = model.recommendForUserSubset(warm, K_OVER)

    # Explode, exclude seen, rank, collect into array — all as one lazy plan
    recs_exp = (
        raw_recs
        .select("user_id", F.explode("recommendations").alias("rec"))
        .select(
            "user_id",
            F.col("rec.book_id").alias("book_id"),
            F.col("rec.rating").alias("pred_score"),
        )
    )

    recs_new = recs_exp.join(train_seen, on=["user_id", "book_id"], how="left_anti")

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
        .withColumn("recs_sorted", F.array_sort("recs_raw"))
        .withColumn("recs_cut",    F.slice("recs_sorted", 1, K_EVAL))
        .withColumn("recs",        F.expr("transform(recs_cut, x -> x.book_id)"))
        .select("user_id", "recs")
    )

    # Write to HDFS — this materialises the plan without collecting to driver
    predictions.write.mode("overwrite").parquet(out_path)
    print(f"[full] {split_name}: predictions saved → {out_path}")
    return n_warm

# ---------------------------------------------------------------------------
# Evaluate val
# ---------------------------------------------------------------------------
print("\n[full] Generating VAL predictions...")
val_preds_path = f"{OUT_DIR}/val_preds_tmp"
generate_and_save_predictions(val_gt, "VAL", val_preds_path)

val_preds = spark.read.parquet(val_preds_path)
val_m = compute_ranking_metrics(val_preds, val_gt, k=K_EVAL)
print(f"[full] VAL  P@100={val_m['precision_at_k']:.4f}  "
      f"MAP@100={val_m['map_at_k']:.4f}  "
      f"NDCG@100={val_m['ndcg_at_k']:.4f}  "
      f"n_users={val_m['n_users_eval']}")

# ---------------------------------------------------------------------------
# Evaluate test
# ---------------------------------------------------------------------------
print("\n[full] Generating TEST predictions...")
test_preds_path = f"{OUT_DIR}/test_preds_tmp"
generate_and_save_predictions(test_gt, "TEST", test_preds_path)

test_preds = spark.read.parquet(test_preds_path)
test_m = compute_ranking_metrics(test_preds, test_gt, k=K_EVAL)
print(f"[full] TEST P@100={test_m['precision_at_k']:.4f}  "
      f"MAP@100={test_m['map_at_k']:.4f}  "
      f"NDCG@100={test_m['ndcg_at_k']:.4f}  "
      f"n_users={test_m['n_users_eval']}")

# ---------------------------------------------------------------------------
# Write summary to HDFS
# ---------------------------------------------------------------------------
summary = [
    "=== Deliverable 3: Explicit-Feedback ALS ===",
    f"rank={BEST_RANK}, regParam={BEST_REG}, maxIter={MAX_ITER}, seed={SEED}",
    "Training filter : rating > 0",
    "Ground truth    : is_read = 1",
    f"Explicit training rows: {n_expl:,}",
    "",
    "--- Validation ---",
    f"  n_users evaluated : {val_m['n_users_eval']:,}",
    f"  Precision@100     : {val_m['precision_at_k']:.6f}",
    f"  MAP@100           : {val_m['map_at_k']:.6f}",
    f"  NDCG@100          : {val_m['ndcg_at_k']:.6f}",
    "",
    "--- Test ---",
    f"  n_users evaluated : {test_m['n_users_eval']:,}",
    f"  Precision@100     : {test_m['precision_at_k']:.6f}",
    f"  MAP@100           : {test_m['map_at_k']:.6f}",
    f"  NDCG@100          : {test_m['ndcg_at_k']:.6f}",
    "",
    "--- vs Popularity baseline (POP_READ, test) ---",
    "  Popularity P@100  : 0.022300",
    "  Popularity MAP@100: 0.031500",
    "  Popularity NDCG100: 0.103600",
    f"  ALS        P@100  : {test_m['precision_at_k']:.6f}",
    f"  ALS        MAP@100: {test_m['map_at_k']:.6f}",
    f"  ALS        NDCG100: {test_m['ndcg_at_k']:.6f}",
]

(spark.createDataFrame([(l,) for l in summary], ["value"])
      .coalesce(1)
      .write.mode("overwrite")
      .text(f"{OUT_DIR}/metrics_summary.txt"))

print(f"\n[full] Summary → {OUT_DIR}/metrics_summary.txt")
spark.stop()
print("[full] Done.")

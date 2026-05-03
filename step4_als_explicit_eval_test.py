"""
step4_als_explicit_eval.py
--------------------------
Deliverable 3 — Explicit ALS: Evaluation only (model already trained).

APPROACH: Instead of recommendForUserSubset (which is slow on 736K users),
we compute scores directly from user and item factor matrices:
  score(u, i) = dot(user_factors[u], item_factors[i])

We use a broadcast-based approach:
  1. Collect ALL item factors to driver (2.36M books × 50 floats = ~470MB — fits)
  2. Broadcast item factors to all executors
  3. For each user partition, compute dot products with all items
  4. Take top-K per user

This mirrors how the popularity baseline worked (broadcast small side,
join with users) and should complete in ~30-60 minutes.

Run (cluster mode):
  spark-submit --master yarn --deploy-mode cluster \
    --num-executors 3 --executor-cores 4 --executor-memory 6G \
    --driver-cores 2 --driver-memory 6G \
    --conf spark.dynamicAllocation.enabled=true \
    --conf spark.dynamicAllocation.maxExecutors=4 \
    --conf spark.shuffle.service.enabled=true \
    --conf spark.sql.shuffle.partitions=200 \
    --conf spark.sql.autoBroadcastJoinThreshold=-1 \
    --conf spark.driver.maxResultSize=4g \
    --conf spark.network.timeout=800s \
    --conf spark.executor.heartbeatInterval=60s \
    --conf spark.task.maxFailures=8 \
    --conf spark.yarn.maxAppAttempts=1 \
    --py-files ~/goodreads_capstone/scripts/eval_helpers.py \
    ~/goodreads_capstone/scripts/step4_als_explicit_eval.py
"""

import sys, os
import numpy as np

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, IntegerType, FloatType, StructType, StructField
from pyspark.ml.recommendation import ALSModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_helpers import compute_ranking_metrics

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE      = "hdfs:///user/at7095_nyu_edu/goodreads"
OUT_DIR   = f"{BASE}/als_explicit/full"
MODEL_DIR = f"{OUT_DIR}/model"
CKPT_DIR  = f"{BASE}/checkpoints"

K_OVER = 200   # candidates before filtering seen items
K_EVAL = 100

# ---------------------------------------------------------------------------
# Spark
# ---------------------------------------------------------------------------
spark = (SparkSession.builder
         .appName("D3_ALS_Explicit_Eval_TEST")
         .config("spark.sql.autoBroadcastJoinThreshold", "-1")
         .config("spark.driver.maxResultSize", "4g")
         .config("spark.sql.shuffle.partitions", "200")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")
spark.sparkContext.setCheckpointDir(CKPT_DIR)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print("[eval] Loading model...")
model = ALSModel.load(MODEL_DIR)
rank  = model.rank
print(f"[eval] Model loaded. rank={rank}")

# ---------------------------------------------------------------------------
# Load precomputed data
# ---------------------------------------------------------------------------
train_seen = spark.read.parquet(f"{OUT_DIR}/train_seen_tmp").cache()
test_gt    = spark.read.parquet(f"{OUT_DIR}/test_gt_tmp").cache()

# ---------------------------------------------------------------------------
# Collect item factors to driver and broadcast
# item_factors: DataFrame with id (int), features (array<float>)
# We collect to numpy array indexed by book_id for fast dot product
# ---------------------------------------------------------------------------
print("[eval] Collecting item factors...")
item_rows = model.itemFactors.collect()
print(f"[eval] Item factors collected: {len(item_rows):,} items")

# Build numpy matrix: item_mat[i] = factor vector for book_id i
# Also build mapping book_id -> row index and back
book_ids  = np.array([r["id"] for r in item_rows], dtype=np.int32)
item_mat  = np.array([r["features"] for r in item_rows], dtype=np.float32)
# item_mat shape: (n_items, rank)

# Broadcast to executors
bc_book_ids = spark.sparkContext.broadcast(book_ids)
bc_item_mat = spark.sparkContext.broadcast(item_mat)

print(f"[eval] Item matrix shape: {item_mat.shape}")
del item_rows  # free driver memory

# ---------------------------------------------------------------------------
# UDF: given a user factor vector, return top-K (book_id, score) pairs
# ---------------------------------------------------------------------------
@F.udf(ArrayType(StructType([
    StructField("book_id", IntegerType(), False),
    StructField("score",   FloatType(),   False),
])))
def top_k_items(user_vec):
    """Score all items for this user, return top-K_OVER as structs."""
    import numpy as np
    bids = bc_book_ids.value
    imat = bc_item_mat.value
    u    = np.array(user_vec, dtype=np.float32)
    scores = imat.dot(u)              # shape: (n_items,)
    # Partial sort: top K_OVER indices
    k = min(K_OVER, len(scores))
    top_idx = np.argpartition(scores, -k)[-k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]  # sort descending
    return [(int(bids[i]), float(scores[i])) for i in top_idx]

# ---------------------------------------------------------------------------
# Generate predictions for a split
# ---------------------------------------------------------------------------
def generate_predictions(gt_df, split_name, preds_path):
    # Warm users: present in both gt and model
    model_user_ids = model.userFactors.select(F.col("id").alias("user_id"))
    warm_users = (gt_df.select("user_id")
                       .join(model_user_ids, on="user_id", how="inner")
                       .distinct())
    n_warm = warm_users.count()
    print(f"[eval] {split_name}: warm users = {n_warm:,}")

    # Join user factors
    user_factors_df = (
        model.userFactors
             .select(F.col("id").alias("user_id"), "features")
             .join(warm_users, on="user_id", how="inner")
    )

    # Apply UDF: get top-K_OVER items per user
    recs_raw = (
        user_factors_df
        .withColumn("top_items", top_k_items(F.col("features")))
        .select("user_id", F.explode("top_items").alias("item"))
        .select(
            "user_id",
            F.col("item.book_id").alias("book_id"),
            F.col("item.score").alias("pred_score"),
        )
    )

    # Exclude training interactions
    recs_new = recs_raw.join(train_seen, on=["user_id", "book_id"], how="left_anti")

    # Checkpoint to break lineage before window
    recs_new = recs_new.checkpoint()

    # Rank and cut to K_EVAL, build array
    w = Window.partitionBy("user_id").orderBy(F.col("pred_score").desc())
    predictions = (
        recs_new
        .withColumn("rank", F.row_number().over(w))
        .filter(F.col("rank") <= K_EVAL)
        .groupBy("user_id")
        .agg(
            F.sort_array(
                F.collect_list(F.struct("rank", "book_id"))
            ).alias("s")
        )
        .withColumn("recs", F.col("s.book_id"))
        .select("user_id", "recs")
    )

    predictions.write.mode("overwrite").parquet(preds_path)
    print(f"[eval] {split_name}: predictions saved → {preds_path}")
    return n_warm

# ---------------------------------------------------------------------------
# Evaluate TEST
# ---------------------------------------------------------------------------
print("\n[eval] === TEST ===")
test_preds_path = f"{OUT_DIR}/test_preds"
generate_predictions(test_gt, "TEST", test_preds_path)

test_preds = spark.read.parquet(test_preds_path)
test_m = compute_ranking_metrics(test_preds, test_gt, k=K_EVAL)
print(f"[eval] TEST P@100={test_m['precision_at_k']:.4f} "
      f"MAP@100={test_m['map_at_k']:.4f} "
      f"NDCG@100={test_m['ndcg_at_k']:.4f} "
      f"n={test_m['n_users_eval']}")

# ---------------------------------------------------------------------------
# Write summary to HDFS
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Write TEST summary to HDFS
# ---------------------------------------------------------------------------
summary = [
    "=== Deliverable 3: Explicit-Feedback ALS — TEST ===",
    "rank=50, regParam=0.1, maxIter=10, seed=1004",
    "Training filter : rating > 0",
    "Ground truth    : is_read = 1",
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
      .text(f"{OUT_DIR}/test_metrics_summary.txt"))

print(f"[eval] TEST summary -> {OUT_DIR}/test_metrics_summary.txt")
spark.stop()

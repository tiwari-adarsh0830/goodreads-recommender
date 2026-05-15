"""
step4_als_explicit_eval_rated.py
--------------------------------
Deliverable 3 alternate evaluation: ground truth = RATED holdout
interactions only (is_read=1 AND rating>0), rather than is_read=1 alone.

Why
---
The default eval (step4_als_explicit_eval_test_small.py) scores explicit
ALS against is_read=1 ground truth. That target covers the 54.3% of
consumption events that have no rating, which the model was never trained
on. The near-zero TEST MAP@100 in the report is partly a property of the
target. This script reruns the eval with ground truth restricted to rated
holdout rows, so the explicit ALS model is judged on the target it was
actually optimizing for. Use both numbers together when interpreting D3.

Shares the saved model from step4_als_explicit_full.py and uses
eval_helpers.compute_ranking_metrics so the comparison stays
apples-to-apples with the rest of the pipeline.

Batching change
---------------
The expensive recommendation step is now processed in warm-user batches
instead of recommending for every warm user in one large Spark job. This
reduces executor/driver pressure and writes intermediate batch prediction
files to HDFS before computing the final metrics.

Optional tuning:
    USER_BATCH_SIZE=5000   # default; lower to 2000 or 1000 if jobs are killed

Run:
    USER_BATCH_SIZE=5000 spark-submit --master yarn --deploy-mode client \
        --num-executors 3 --executor-cores 4 --executor-memory 6G \
        --driver-memory 2G \
        --conf spark.sql.autoBroadcastJoinThreshold=-1 \
        --py-files ~/goodreads_capstone/scripts/eval_helpers.py \
        ~/goodreads_capstone/scripts/step4_als_explicit_eval_rated.py
"""
import os
import sys
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.storagelevel import StorageLevel
from pyspark.ml.recommendation import ALSModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_helpers import compute_ranking_metrics

NETID     = os.environ.get("NETID", "ss21040")
BASE      = f"hdfs:///user/{NETID}_nyu_edu/goodreads"
OUT_DIR   = f"{BASE}/als_explicit/full"
MODEL_DIR = f"{OUT_DIR}/model"
CKPT_DIR  = f"{BASE}/checkpoints"
K_OVER    = 1000
K_EVAL    = 100

# Number of warm users handled per recommendForUserSubset call.
# If the YARN job is still getting killed, run with USER_BATCH_SIZE=2000 or 1000.
USER_BATCH_SIZE = int(os.environ.get("USER_BATCH_SIZE", "5000"))

spark = (SparkSession.builder
         .appName("D3_ALS_Explicit_Eval_RATED_BATCHED")
         .config("spark.sql.autoBroadcastJoinThreshold", "-1")
         .config("spark.driver.maxResultSize", "4g")
         .config("spark.sql.shuffle.partitions", "200")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")
spark.sparkContext.setCheckpointDir(CKPT_DIR)

print("[eval-rated] Loading model...")
model = ALSModel.load(MODEL_DIR)
print(f"[eval-rated] Model loaded. rank={model.rank}")
print(f"[eval-rated] USER_BATCH_SIZE={USER_BATCH_SIZE:,}")

val_small   = spark.read.parquet(f"{BASE}/val_small.parquet")
test_small  = spark.read.parquet(f"{BASE}/test_small.parquet")
train_small = spark.read.parquet(f"{BASE}/train_small.parquet")

# Rated-only ground truth: is_read=1 AND rating>0.
val_gt_rated = (val_small.filter((F.col("is_read") == 1) & (F.col("rating") > 0))
                         .groupBy("user_id")
                         .agg(F.collect_list("book_id").alias("ground_truth"))
                         .persist(StorageLevel.MEMORY_AND_DISK))

test_gt_rated = (test_small.filter((F.col("is_read") == 1) & (F.col("rating") > 0))
                           .groupBy("user_id")
                           .agg(F.collect_list("book_id").alias("ground_truth"))
                           .persist(StorageLevel.MEMORY_AND_DISK))

# Block books the user already rated in train -- the model only knows rated
# books anyway, so this is the meaningful seen-set for D3.
train_seen_rated = (train_small.filter(F.col("rating") > 0)
                               .select("user_id", "book_id")
                               .distinct()
                               .repartition("user_id")
                               .persist(StorageLevel.MEMORY_AND_DISK))

model_users = (model.userFactors
                    .select(F.col("id").alias("user_id"))
                    .persist(StorageLevel.MEMORY_AND_DISK))

# Materialize shared inputs once so repeated batch jobs do not rebuild them.
val_gt_rated.count()
test_gt_rated.count()
train_seen_rated.count()
model_users.count()


def _build_preds_from_recs(recs_df):
    """Remove already-rated train books and keep the top K_EVAL recs per user."""
    recs_new = recs_df.join(
        train_seen_rated,
        on=["user_id", "book_id"],
        how="left_anti"
    )

    w = Window.partitionBy("user_id").orderBy(F.col("pred_score").desc())
    return (
        recs_new
        .withColumn("rank", F.row_number().over(w))
        .filter(F.col("rank") <= K_EVAL)
        .groupBy("user_id")
        .agg(F.sort_array(F.collect_list(F.struct("rank", "book_id"))).alias("s"))
        .withColumn("recs", F.col("s.book_id"))
        .select("user_id", "recs")
    )


def eval_split(gt_df, split_name):
    warm = (gt_df.select("user_id")
                 .join(model_users, on="user_id", how="inner")
                 .distinct())

    n_warm = warm.count()
    print(f"[eval-rated] {split_name} warm users (rated GT): {n_warm:,}")
    if n_warm == 0:
        return {"n_users_eval": 0, "precision_at_k": 0.0,
                "map_at_k": 0.0, "ndcg_at_k": 0.0}

    n_batches = max(1, (n_warm + USER_BATCH_SIZE - 1) // USER_BATCH_SIZE)
    print(f"[eval-rated] {split_name} processing {n_batches:,} user batches")

    batch_preds_path = f"{OUT_DIR}/{split_name.lower()}_small_preds_rated_batches"
    final_preds_path = f"{OUT_DIR}/{split_name.lower()}_small_preds_rated"

    # Hash-based batch assignment avoids a full global sort over all warm users.
    warm_batched = (warm
        .withColumn("batch_id", F.pmod(F.hash(F.col("user_id")), F.lit(n_batches)))
        .repartition(n_batches, "batch_id")
        .persist(StorageLevel.MEMORY_AND_DISK)
    )
    warm_batched.count()

    wrote_any_batch = False

    for batch_id in range(n_batches):
        batch_users = (warm_batched
                       .filter(F.col("batch_id") == batch_id)
                       .select("user_id"))

        n_batch_users = batch_users.count()
        if n_batch_users == 0:
            print(f"[eval-rated] {split_name} batch {batch_id + 1}/{n_batches}: empty, skipping")
            continue

        print(f"[eval-rated] {split_name} batch {batch_id + 1}/{n_batches}: "
              f"{n_batch_users:,} users")

        recs = (model.recommendForUserSubset(batch_users, K_OVER)
                     .select("user_id", F.explode("recommendations").alias("rec"))
                     .select("user_id",
                             F.col("rec.book_id").alias("book_id"),
                             F.col("rec.rating").alias("pred_score")))

        preds_batch = _build_preds_from_recs(recs)

        write_mode = "overwrite" if not wrote_any_batch else "append"
        (preds_batch
            .write
            .mode(write_mode)
            .parquet(batch_preds_path))

        wrote_any_batch = True
        print(f"[eval-rated] {split_name} batch {batch_id + 1}/{n_batches}: written")

    warm_batched.unpersist()

    if not wrote_any_batch:
        return {"n_users_eval": 0, "precision_at_k": 0.0,
                "map_at_k": 0.0, "ndcg_at_k": 0.0}

    preds = spark.read.parquet(batch_preds_path).persist(StorageLevel.MEMORY_AND_DISK)
    n_pred_users = preds.count()
    print(f"[eval-rated] {split_name} prediction users after batching: {n_pred_users:,}")

    # Keep the original output path name for downstream/report compatibility.
    preds.write.mode("overwrite").parquet(final_preds_path)

    m = compute_ranking_metrics(preds, gt_df, k=K_EVAL)
    print(f"[eval-rated] {split_name} P@100={m['precision_at_k']:.4f} "
          f"MAP@100={m['map_at_k']:.4f} NDCG@100={m['ndcg_at_k']:.4f} "
          f"n={m['n_users_eval']}")

    preds.unpersist()
    return m


val_m  = eval_split(val_gt_rated,  "VAL")
test_m = eval_split(test_gt_rated, "TEST")

summary = [
    "=== Deliverable 3 (rated-only GT): Explicit ALS - Small Split ===",
    "rank=50, regParam=0.1, maxIter=10, seed=1004",
    f"User batch size : {USER_BATCH_SIZE:,}",
    "Training filter : rating > 0",
    "Ground truth    : is_read = 1 AND rating > 0  (RATED HOLDOUT ONLY)",
    "Seen filter     : training rows with rating > 0",
    "Recommendation  : warm users processed in batches",
    "",
    "--- Validation (small split, rated GT) ---",
    f"  n_users evaluated : {val_m['n_users_eval']:,}",
    f"  Precision@100     : {val_m['precision_at_k']:.6f}",
    f"  MAP@100           : {val_m['map_at_k']:.6f}",
    f"  NDCG@100          : {val_m['ndcg_at_k']:.6f}",
    "",
    "--- Test (small split, rated GT) ---",
    f"  n_users evaluated : {test_m['n_users_eval']:,}",
    f"  Precision@100     : {test_m['precision_at_k']:.6f}",
    f"  MAP@100           : {test_m['map_at_k']:.6f}",
    f"  NDCG@100          : {test_m['ndcg_at_k']:.6f}",
    "",
    "Interpretation note: compare against the default is_read=1 numbers in",
    "step4_val_metrics.txt / step4 test_small_metrics_summary.txt. A lift",
    "here without a corresponding lift on is_read=1 means D3 is recovering",
    "preference order on rated items but missing the unrated consumption",
    "events the report's main eval target requires.",
]

(spark.createDataFrame([(l,) for l in summary], ["value"])
      .coalesce(1)
      .write.mode("overwrite")
      .text(f"{OUT_DIR}/rated_gt_metrics_summary.txt"))

print(f"[eval-rated] Done -> {OUT_DIR}/rated_gt_metrics_summary.txt")

val_gt_rated.unpersist()
test_gt_rated.unpersist()
train_seen_rated.unpersist()
model_users.unpersist()
spark.stop()

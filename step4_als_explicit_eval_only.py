import sys, os
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALSModel
sys.path.insert(0, '/home/at7095_nyu_edu')
from eval_helpers import compute_ranking_metrics

BASE      = "hdfs:///user/at7095_nyu_edu/goodreads"
MODEL_DIR = f"{BASE}/als_explicit/full/model"
OUT_DIR   = f"{BASE}/als_explicit/fixed"
CKPT_DIR  = f"{BASE}/checkpoints"
K_OVER    = 500
K_EVAL    = 100

spark = (SparkSession.builder
         .appName("D3_ALS_Explicit_EvalOnly_v2")
         .config("spark.sql.autoBroadcastJoinThreshold", "-1")
         .config("spark.driver.maxResultSize", "4g")
         .config("spark.sql.shuffle.partitions", "4000")
         .getOrCreate())
spark.sparkContext.setCheckpointDir(CKPT_DIR)
spark.sparkContext.setLogLevel("WARN")

print("[D3] Loading model...")
model = ALSModel.load(MODEL_DIR)
print(f"[D3] Model loaded. rank={model.rank}")

train_raw = spark.read.parquet(f"{BASE}/train.parquet")
val_raw   = spark.read.parquet(f"{BASE}/val.parquet")
test_raw  = spark.read.parquet(f"{BASE}/test.parquet")

train_seen = (train_raw
              .filter(F.col("rating") > 0)
              .select("user_id", "book_id")
              .distinct()
              .cache())
train_seen.count()
print("[D3] train_seen cached.")

def make_gt(df):
    return (df
            .filter(F.col("is_read") == 1)
            .groupBy("user_id")
            .agg(F.collect_set("book_id").alias("ground_truth")))

val_gt  = make_gt(val_raw).cache()
test_gt = make_gt(test_raw).cache()
print(f"[D3] Val GT users:  {val_gt.count():,}")
print(f"[D3] Test GT users: {test_gt.count():,}")

model_users = model.userFactors.select(F.col("id").alias("user_id"))

def evaluate(gt_df, split_name):
    warm = (gt_df.select("user_id")
                 .join(model_users, "user_id", "inner")
                 .distinct())
    n_warm = warm.count()
    print(f"[D3] {split_name} warm users: {n_warm:,}")

    raw_recs = model.recommendForUserSubset(warm, K_OVER)
    recs_exp = (raw_recs
                .select("user_id", F.explode("recommendations").alias("rec"))
                .select("user_id",
                        F.col("rec.book_id").alias("book_id"),
                        F.col("rec.rating").alias("pred_score")))

    recs_filtered = recs_exp.join(train_seen, ["user_id", "book_id"], "left_anti")

    w = Window.partitionBy("user_id").orderBy(F.col("pred_score").desc())
    preds = (recs_filtered
             .withColumn("rank", F.row_number().over(w))
             .filter(F.col("rank") <= K_EVAL)
             .groupBy("user_id")
             .agg(F.sort_array(
                 F.collect_list(F.struct("rank", "book_id"))
             ).alias("s"))
             .withColumn("recs", F.col("s.book_id"))
             .select("user_id", "recs"))

    # Write to HDFS immediately — keeps job active, no idle periods
    preds_path = f"{OUT_DIR}/{split_name.lower()}_preds_full"
    preds.write.mode("overwrite").parquet(preds_path)
    print(f"[D3] {split_name} predictions written to HDFS.")

    # Read back and compute metrics
    preds_saved = spark.read.parquet(preds_path)
    m = compute_ranking_metrics(preds_saved, gt_df, k=K_EVAL)
    print(f"[D3] {split_name} P@100={m['precision_at_k']:.4f} "
          f"MAP@100={m['map_at_k']:.4f} "
          f"NDCG@100={m['ndcg_at_k']:.4f} "
          f"n={m['n_users_eval']:,}")

    # Write metrics immediately after computing
    (spark.createDataFrame(
        [(f"{split_name} P@100={m['precision_at_k']:.6f} "
          f"MAP@100={m['map_at_k']:.6f} "
          f"NDCG@100={m['ndcg_at_k']:.6f} "
          f"n={m['n_users_eval']}",)], ["value"])
     .coalesce(1).write.mode("overwrite")
     .text(f"{OUT_DIR}/{split_name.lower()}_metrics_full.txt"))
    print(f"[D3] {split_name} metrics saved to HDFS.")

    return m

val_m  = evaluate(val_gt,  "VAL")
test_m = evaluate(test_gt, "TEST")

summary = [
    "=== D3 Explicit ALS - FULL SCALE ===",
    f"rank={model.rank} regParam=0.1 maxIter=10 seed=1004",
    "train filter : rating > 0",
    "train_seen   : rating > 0 only",
    "ground truth : is_read == 1",
    f"K_OVER={K_OVER} K_EVAL={K_EVAL}",
    "",
    f"VAL  P@100={val_m['precision_at_k']:.6f} MAP@100={val_m['map_at_k']:.6f} NDCG@100={val_m['ndcg_at_k']:.6f} n={val_m['n_users_eval']}",
    f"TEST P@100={test_m['precision_at_k']:.6f} MAP@100={test_m['map_at_k']:.6f} NDCG@100={test_m['ndcg_at_k']:.6f} n={test_m['n_users_eval']}",
]
(spark.createDataFrame([(l,) for l in summary], ["value"])
      .coalesce(1).write.mode("overwrite")
      .text(f"{OUT_DIR}/metrics_summary_full.txt"))

print("[D3] Done.")
spark.stop()

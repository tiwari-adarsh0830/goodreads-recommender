import sys, os
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALSModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_helpers import compute_ranking_metrics

BASE      = "hdfs:///user/at7095_nyu_edu/goodreads"
OUT_DIR   = f"{BASE}/als_implicit/full"
MODEL_DIR = f"{OUT_DIR}/model"
CKPT_DIR  = f"{BASE}/checkpoints"
K_OVER    = 1000
K_EVAL    = 100

spark = (SparkSession.builder
         .appName("D4_ALS_Implicit_Eval_SMALL")
         .config("spark.sql.autoBroadcastJoinThreshold", "-1")
         .config("spark.driver.maxResultSize", "4g")
         .config("spark.sql.shuffle.partitions", "200")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")
spark.sparkContext.setCheckpointDir(CKPT_DIR)

print("[eval] Loading model...")
model = ALSModel.load(MODEL_DIR)
print(f"[eval] Model loaded. rank={model.rank}")

val_small  = spark.read.parquet(f"{BASE}/val_small.parquet")
train_small = spark.read.parquet(f"{BASE}/train_small.parquet")

val_gt = (val_small.filter(F.col("is_read") == 1)
                   .groupBy("user_id")
                   .agg(F.collect_list("book_id").alias("ground_truth")))

train_seen = train_small.select("user_id", "book_id").distinct()

model_users = model.userFactors.select(F.col("id").alias("user_id"))
warm_users  = (val_gt.select("user_id")
                     .join(model_users, on="user_id", how="inner")
                     .distinct())

n_warm = warm_users.count()
print(f"[eval] Warm users: {n_warm:,}")

recs = (model.recommendForUserSubset(warm_users, K_OVER)
             .select("user_id", F.explode("recommendations").alias("rec"))
             .select("user_id",
                     F.col("rec.book_id").alias("book_id"),
                     F.col("rec.rating").alias("pred_score")))

recs_new = recs.join(train_seen, on=["user_id", "book_id"], how="left_anti")

w = Window.partitionBy("user_id").orderBy(F.col("pred_score").desc())
val_preds = (
    recs_new
    .withColumn("rank", F.row_number().over(w))
    .filter(F.col("rank") <= K_EVAL)
    .groupBy("user_id")
    .agg(F.sort_array(F.collect_list(F.struct("rank", "book_id"))).alias("s"))
    .withColumn("recs", F.col("s.book_id"))
    .select("user_id", "recs")
)

val_preds.write.mode("overwrite").parquet(f"{OUT_DIR}/val_small_preds")

val_m = compute_ranking_metrics(val_preds, val_gt, k=K_EVAL)
print(f"[eval] VAL_SMALL P@100={val_m['precision_at_k']:.4f} MAP@100={val_m['map_at_k']:.4f} NDCG@100={val_m['ndcg_at_k']:.4f} n={val_m['n_users_eval']}")

summary = [
    "=== Deliverable 4: Implicit-Feedback ALS - VAL (small split) ===",
    "rank=50, regParam=0.1, maxIter=10, alpha=40.0, seed=1004",
    "Training data  : ALL interactions (rating>=0)",
    "Confidence     : 1 + 40 * is_read",
    "Ground truth   : is_read = 1",
    "",
    "--- Validation (small split) ---",
    f"  n_users evaluated : {val_m['n_users_eval']:,}",
    f"  Precision@100     : {val_m['precision_at_k']:.6f}",
    f"  MAP@100           : {val_m['map_at_k']:.6f}",
    f"  NDCG@100          : {val_m['ndcg_at_k']:.6f}",
]

(spark.createDataFrame([(l,) for l in summary], ["value"])
      .coalesce(1)
      .write.mode("overwrite")
      .text(f"{OUT_DIR}/val_small_metrics_summary.txt"))

print(f"[eval] Done -> {OUT_DIR}/val_small_metrics_summary.txt")
spark.stop()

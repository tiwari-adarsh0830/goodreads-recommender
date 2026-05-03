import sys, os
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALSModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_helpers import compute_ranking_metrics

BASE       = "hdfs:///user/at7095_nyu_edu/goodreads"
OUT_DIR    = f"{BASE}/als_explicit/full"
MODEL_DIR  = f"{OUT_DIR}/model"
CKPT_DIR   = f"{BASE}/checkpoints"
BATCH_SIZE = 2000
K_OVER     = 1000
K_EVAL     = 100

spark = (SparkSession.builder
         .appName("D3_ALS_Explicit_Eval_VAL")
         .config("spark.sql.autoBroadcastJoinThreshold", "-1")
         .config("spark.driver.maxResultSize", "4g")
         .config("spark.sql.shuffle.partitions", "400")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")
spark.sparkContext.setCheckpointDir(CKPT_DIR)

print("[eval] Loading model...")
model = ALSModel.load(MODEL_DIR)
print(f"[eval] Model loaded. rank={model.rank}")

train_seen = spark.read.parquet(f"{OUT_DIR}/train_seen_tmp")
val_gt     = spark.read.parquet(f"{OUT_DIR}/val_gt_tmp")

model_users = model.userFactors.select(F.col("id").alias("user_id"))
warm_users  = (val_gt.select("user_id")
                     .join(model_users, on="user_id", how="inner")
                     .distinct()
                     .cache())

n_warm = warm_users.count()
print(f"[eval] Warm val users: {n_warm:,}")

user_id_list = [r["user_id"] for r in warm_users.select("user_id").collect()]
n_batches = (len(user_id_list) + BATCH_SIZE - 1) // BATCH_SIZE
print(f"[eval] Processing {n_batches} batches of {BATCH_SIZE} users")

val_preds_path = f"{OUT_DIR}/val_preds"
try:
    fs = spark.sparkContext._jvm.org.apache.hadoop.fs.FileSystem.get(spark.sparkContext._jsc.hadoopConfiguration())
    fs.delete(spark.sparkContext._jvm.org.apache.hadoop.fs.Path(val_preds_path), True)
    print("[eval] Cleared previous val_preds")
except:
    pass

for batch_idx in range(n_batches):
    batch_ids = user_id_list[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]
    batch_df  = spark.createDataFrame([(uid,) for uid in batch_ids], ["user_id"])

    recs = (model.recommendForUserSubset(batch_df, K_OVER)
                 .select("user_id", F.explode("recommendations").alias("rec"))
                 .select("user_id",
                         F.col("rec.book_id").alias("book_id"),
                         F.col("rec.rating").alias("pred_score")))

    recs_new = recs.join(train_seen, on=["user_id", "book_id"], how="left_anti")

    w = Window.partitionBy("user_id").orderBy(F.col("pred_score").desc())
    batch_preds = (
        recs_new
        .withColumn("rank", F.row_number().over(w))
        .filter(F.col("rank") <= K_EVAL)
        .groupBy("user_id")
        .agg(F.sort_array(F.collect_list(F.struct("rank", "book_id"))).alias("s"))
        .withColumn("recs", F.col("s.book_id"))
        .select("user_id", "recs")
    )

    batch_preds.write.mode("append").parquet(val_preds_path)

    if batch_idx % 10 == 0:
        print(f"[eval] Batch {batch_idx+1}/{n_batches} done")

print("[eval] All batches complete.")

val_preds = spark.read.parquet(val_preds_path)
val_m = compute_ranking_metrics(val_preds, val_gt, k=K_EVAL)
print(f"[eval] VAL  P@100={val_m['precision_at_k']:.4f} MAP@100={val_m['map_at_k']:.4f} NDCG@100={val_m['ndcg_at_k']:.4f} n={val_m['n_users_eval']}")

summary = [
    "=== Deliverable 3: Explicit-Feedback ALS - VAL ===",
    "rank=50, regParam=0.1, maxIter=10, seed=1004",
    "Training filter : rating > 0",
    "Ground truth    : is_read = 1",
    "",
    "--- Validation ---",
    f"  n_users evaluated : {val_m['n_users_eval']:,}",
    f"  Precision@100     : {val_m['precision_at_k']:.6f}",
    f"  MAP@100           : {val_m['map_at_k']:.6f}",
    f"  NDCG@100          : {val_m['ndcg_at_k']:.6f}",
]

(spark.createDataFrame([(l,) for l in summary], ["value"])
      .coalesce(1)
      .write.mode("overwrite")
      .text(f"{OUT_DIR}/val_metrics_summary.txt"))

print(f"[eval] VAL summary -> {OUT_DIR}/val_metrics_summary.txt")
spark.stop()

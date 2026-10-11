import sys, os
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALSModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_helpers import compute_ranking_metrics

BASE      = f"hdfs:///user/{os.environ.get('NETID', 'your_netid')}_nyu_edu/goodreads"
OUT_DIR   = f"{BASE}/als_implicit_v2/full"
MODEL_DIR = f"{OUT_DIR}/model"
CKPT_DIR  = f"{BASE}/checkpoints"
K_OVER    = 1000
K_EVAL    = 100

spark = (SparkSession.builder
         .appName("D4v2_ALS_Implicit_Eval_SMALL")
         .config("spark.sql.autoBroadcastJoinThreshold", "-1")
         .config("spark.driver.maxResultSize", "4g")
         .config("spark.sql.shuffle.partitions", "200")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")
spark.sparkContext.setCheckpointDir(CKPT_DIR)

print("[eval] Loading model...")
model = ALSModel.load(MODEL_DIR)
print(f"[eval] Model loaded. rank={model.rank}")

train_small = spark.read.parquet(f"{BASE}/train_small.parquet")
val_small   = spark.read.parquet(f"{BASE}/val_small.parquet")
test_small  = spark.read.parquet(f"{BASE}/test_small.parquet")

train_seen = train_small.select("user_id", "book_id").distinct()

val_gt = (val_small.filter(F.col("is_read") == 1)
                   .groupBy("user_id")
                   .agg(F.collect_list("book_id").alias("ground_truth")))

test_gt = (test_small.filter(F.col("is_read") == 1)
                     .groupBy("user_id")
                     .agg(F.collect_list("book_id").alias("ground_truth")))

model_users = model.userFactors.select(F.col("id").alias("user_id"))

def eval_split(gt_df, split_name):
    warm = (gt_df.select("user_id")
                 .join(model_users, on="user_id", how="inner")
                 .distinct())
    n_warm = warm.count()
    print(f"[eval] {split_name} warm users: {n_warm:,}")

    recs = (model.recommendForUserSubset(warm, K_OVER)
                 .select("user_id", F.explode("recommendations").alias("rec"))
                 .select("user_id",
                         F.col("rec.book_id").alias("book_id"),
                         F.col("rec.rating").alias("pred_score")))

    recs_new = recs.join(train_seen, on=["user_id", "book_id"], how="left_anti")

    w = Window.partitionBy("user_id").orderBy(F.col("pred_score").desc())
    preds = (
        recs_new
        .withColumn("rank", F.row_number().over(w))
        .filter(F.col("rank") <= K_EVAL)
        .groupBy("user_id")
        .agg(F.sort_array(F.collect_list(F.struct("rank", "book_id"))).alias("s"))
        .withColumn("recs", F.col("s.book_id"))
        .select("user_id", "recs")
    )

    preds.write.mode("overwrite").parquet(f"{OUT_DIR}/{split_name.lower()}_small_preds")
    m = compute_ranking_metrics(preds, gt_df, k=K_EVAL)
    print(f"[eval] {split_name} P@100={m['precision_at_k']:.4f} MAP@100={m['map_at_k']:.4f} NDCG@100={m['ndcg_at_k']:.4f} n={m['n_users_eval']}")
    return m

val_m  = eval_split(val_gt,  "VAL")
test_m = eval_split(test_gt, "TEST")

summary = [
    "=== Deliverable 4v2: Implicit ALS (is_read + is_reviewed) - Small Split ===",
    "rank=50, regParam=0.1, maxIter=10, alpha=40.0, seed=1004",
    "Signal: is_read + 0.5 * is_reviewed",
    "Ground truth: is_read = 1",
    "",
    "--- Validation (small split) ---",
    f"  n_users evaluated : {val_m['n_users_eval']:,}",
    f"  Precision@100     : {val_m['precision_at_k']:.6f}",
    f"  MAP@100           : {val_m['map_at_k']:.6f}",
    f"  NDCG@100          : {val_m['ndcg_at_k']:.6f}",
    "",
    "--- Test (small split) ---",
    f"  n_users evaluated : {test_m['n_users_eval']:,}",
    f"  Precision@100     : {test_m['precision_at_k']:.6f}",
    f"  MAP@100           : {test_m['map_at_k']:.6f}",
    f"  NDCG@100          : {test_m['ndcg_at_k']:.6f}",
]

(spark.createDataFrame([(l,) for l in summary], ["value"])
      .coalesce(1)
      .write.mode("overwrite")
      .text(f"{OUT_DIR}/metrics_summary.txt"))

print(f"[eval] Done -> {OUT_DIR}/metrics_summary.txt")
spark.stop()

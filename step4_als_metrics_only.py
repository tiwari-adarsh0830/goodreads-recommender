import sys, os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_helpers import compute_ranking_metrics

BASE     = f"hdfs:///user/{os.environ.get('NETID', 'your_netid')}_nyu_edu/goodreads"
OUT_DIR  = f"{BASE}/als_explicit/full"

spark = (SparkSession.builder
         .appName("D3_ALS_Metrics_Only")
         .config("spark.sql.autoBroadcastJoinThreshold", "-1")
         .config("spark.driver.maxResultSize", "4g")
         .config("spark.sql.shuffle.partitions", "200")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

val_preds = spark.read.parquet(f"{OUT_DIR}/val_preds")
val_gt    = spark.read.parquet(f"{OUT_DIR}/val_gt_tmp")

val_m = compute_ranking_metrics(val_preds, val_gt, k=100)
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

print(f"[eval] Done -> {OUT_DIR}/val_metrics_summary.txt")
spark.stop()

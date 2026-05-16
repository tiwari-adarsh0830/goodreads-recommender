import sys
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALSModel

sys.path.insert(0, '/home/at7095_nyu_edu')
from eval_helpers import compute_ranking_metrics

BASE     = "hdfs:///user/at7095_nyu_edu/goodreads"
CKPT_DIR = f"{BASE}/checkpoints"
K_OVER   = 500
K_EVAL   = 100
SAMPLE   = 0.03
SEED     = 1004

spark = (SparkSession.builder
         .appName("D3_3pct_Eval")
         .config("spark.sql.autoBroadcastJoinThreshold", "-1")
         .config("spark.driver.maxResultSize", "4g")
         .config("spark.sql.shuffle.partitions", "400")
         .getOrCreate())
spark.sparkContext.setCheckpointDir(CKPT_DIR)
spark.sparkContext.setLogLevel("WARN")

train_raw = spark.read.parquet(f"{BASE}/train.parquet")
val_raw   = spark.read.parquet(f"{BASE}/val.parquet")
test_raw  = spark.read.parquet(f"{BASE}/test.parquet")

# Sample 3% of users from full user pool
all_users = train_raw.select("user_id").distinct()
sample_users = all_users.sample(False, SAMPLE, seed=SEED).cache()
n_sample = sample_users.count()
print(f"[eval] Sampled {n_sample:,} users (3%)")

val_3pct   = val_raw.join(sample_users,   "user_id", "inner")
test_3pct  = test_raw.join(sample_users,  "user_id", "inner")
train_3pct = train_raw.join(sample_users, "user_id", "inner")

# D3 train_seen: rated rows only
train_seen = (train_3pct
              .filter(F.col("rating") > 0)
              .select("user_id", "book_id")
              .distinct()
              .cache())
train_seen.count()
print("[eval] train_seen cached (rated only).")

def make_gt(df):
    return (df.filter(F.col("is_read") == 1)
              .groupBy("user_id")
              .agg(F.collect_set("book_id").alias("ground_truth")))

val_gt  = make_gt(val_3pct).cache()
test_gt = make_gt(test_3pct).cache()
print(f"[eval] Val GT users:  {val_gt.count():,}")
print(f"[eval] Test GT users: {test_gt.count():,}")

print("\n=== D3 Explicit ALS ===")
model = ALSModel.load(f"{BASE}/als_explicit/full/model")
model_users = model.userFactors.select(F.col("id").alias("user_id"))

def evaluate(gt_df, split_name):
    warm = gt_df.select("user_id").join(model_users, "user_id", "inner").distinct()
    n_warm = warm.count()
    print(f"[eval] {split_name} warm users: {n_warm:,}")

    recs = (model.recommendForUserSubset(warm, K_OVER)
                 .select("user_id", F.explode("recommendations").alias("rec"))
                 .select("user_id",
                         F.col("rec.book_id").alias("book_id"),
                         F.col("rec.rating").alias("pred_score")))

    recs_filtered = recs.join(train_seen, ["user_id", "book_id"], "left_anti")

    w = Window.partitionBy("user_id").orderBy(F.col("pred_score").desc())
    preds = (recs_filtered
             .withColumn("rank", F.row_number().over(w))
             .filter(F.col("rank") <= K_EVAL)
             .groupBy("user_id")
             .agg(F.sort_array(F.collect_list(F.struct("rank", "book_id"))).alias("s"))
             .withColumn("recs", F.col("s.book_id"))
             .select("user_id", "recs"))

    preds_path = f"{BASE}/als_explicit/{split_name.lower()}_3pct_preds"
    preds.write.mode("overwrite").parquet(preds_path)
    print(f"[eval] {split_name} predictions written.")

    preds_saved = spark.read.parquet(preds_path)
    m = compute_ranking_metrics(preds_saved, gt_df, k=K_EVAL)
    print(f"[eval] {split_name} P@100={m['precision_at_k']:.4f} "
          f"MAP@100={m['map_at_k']:.4f} "
          f"NDCG@100={m['ndcg_at_k']:.4f} "
          f"n={m['n_users_eval']:,}")

    (spark.createDataFrame(
        [(f"{split_name} P@100={m['precision_at_k']:.6f} "
          f"MAP@100={m['map_at_k']:.6f} "
          f"NDCG@100={m['ndcg_at_k']:.6f} "
          f"n={m['n_users_eval']}",)], ["value"])
     .coalesce(1).write.mode("overwrite")
     .text(f"{BASE}/als_explicit/{split_name.lower()}_3pct_metrics.txt"))
    print(f"[eval] {split_name} metrics saved.")
    return m

val_m  = evaluate(val_gt,  "VAL")
test_m = evaluate(test_gt, "TEST")

print(f"\nD3 VAL  MAP@100={val_m['map_at_k']:.6f}")
print(f"D3 TEST MAP@100={test_m['map_at_k']:.6f}")

spark.stop()
print("[eval] Done.")

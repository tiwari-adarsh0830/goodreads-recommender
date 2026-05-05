"""
step4_als_explicit_fixed.py
Deliverable 3 — Explicit ALS, fixed.
- K_OVER raised to 2000 to ensure enough candidates survive left_anti join
- train_seen restricted to rating>0 rows only (books the model actually knows)
- Diagnostic prints added to confirm rec counts before eval
"""

import sys, os
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS

sys.path.insert(0, '/home/at7095_nyu_edu')
from eval_helpers import compute_ranking_metrics

BASE      = "hdfs:///user/at7095_nyu_edu/goodreads"
SMALL     = f"{BASE}"
OUT_DIR   = f"{BASE}/als_explicit/fixed"
CKPT_DIR  = f"{BASE}/checkpoints"

BEST_RANK = 50
BEST_REG  = 0.1
MAX_ITER  = 10
SEED      = 1004
K_OVER    = 2000   # <-- raised: model knows ~800K books, need enough candidates
K_EVAL    = 100

spark = (SparkSession.builder
         .appName("D3_ALS_Explicit_Fixed")
         .config("spark.sql.autoBroadcastJoinThreshold", "-1")
         .config("spark.driver.maxResultSize", "4g")
         .config("spark.sql.shuffle.partitions", "400")
         .getOrCreate())
spark.sparkContext.setCheckpointDir(CKPT_DIR)
spark.sparkContext.setLogLevel("WARN")

# Use small split for speed — same model, reproducible
train_raw = spark.read.parquet(f"{SMALL}/train_small.parquet")
val_raw   = spark.read.parquet(f"{SMALL}/val_small.parquet")
test_raw  = spark.read.parquet(f"{SMALL}/test_small.parquet")

# Explicit training rows only
train_explicit = (
    train_raw
    .filter(F.col("rating") > 0)
    .select("user_id", "book_id", F.col("rating").cast("float"))
    .cache()
)
n_expl = train_explicit.count()
print(f"[D3] Explicit training rows: {n_expl:,}")

# train_seen: only rated books (what the model actually knows)
# Blocking unrated books is wrong — model can't recommend them anyway
train_seen = train_explicit.select("user_id", "book_id").distinct()

# Ground truth: is_read=1 in val/test
def make_gt(df):
    return (df
            .filter(F.col("is_read") == 1)
            .groupBy("user_id")
            .agg(F.collect_set("book_id").alias("ground_truth")))

val_gt  = make_gt(val_raw).cache()
test_gt = make_gt(test_raw).cache()
print(f"[D3] Val GT users:  {val_gt.count():,}")
print(f"[D3] Test GT users: {test_gt.count():,}")

# Train
print(f"[D3] Training ALS rank={BEST_RANK} regParam={BEST_REG}...")
als = ALS(
    rank=BEST_RANK, maxIter=MAX_ITER, regParam=BEST_REG,
    userCol="user_id", itemCol="book_id", ratingCol="rating",
    implicitPrefs=False, coldStartStrategy="drop",
    seed=SEED, nonnegative=False,
)
model = als.fit(train_explicit)
print("[D3] Training complete.")

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

    # Exclude only books the user rated in training
    recs_filtered = recs_exp.join(train_seen, ["user_id", "book_id"], "left_anti")

    # Diagnostic: how many candidates survive per user on average?
    avg_candidates = (recs_filtered
                      .groupBy("user_id")
                      .agg(F.count("*").alias("n"))
                      .agg(F.mean("n"), F.min("n"), F.max("n"))
                      .collect()[0])
    print(f"[D3] {split_name} candidates after filter — "
          f"mean={avg_candidates[0]:.1f} min={avg_candidates[1]} max={avg_candidates[2]}")

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

    # Diagnostic: how many users have full 100 recs?
    full_recs = preds.filter(F.size("recs") == K_EVAL).count()
    print(f"[D3] {split_name} users with full {K_EVAL} recs: {full_recs:,} / {n_warm:,}")

    m = compute_ranking_metrics(preds, gt_df, k=K_EVAL)
    print(f"[D3] {split_name} P@100={m['precision_at_k']:.4f} "
          f"MAP@100={m['map_at_k']:.4f} "
          f"NDCG@100={m['ndcg_at_k']:.4f} "
          f"n={m['n_users_eval']:,}")
    return m

val_m  = evaluate(val_gt,  "VAL")
test_m = evaluate(test_gt, "TEST")

summary = [
    "=== D3 Explicit ALS FIXED ===",
    f"rank={BEST_RANK} regParam={BEST_REG} maxIter={MAX_ITER} seed={SEED}",
    "train filter : rating > 0",
    "train_seen   : rating > 0 only (model cannot recommend unrated books)",
    "ground truth : is_read == 1",
    f"K_OVER={K_OVER} K_EVAL={K_EVAL}",
    "",
    f"VAL  P@100={val_m['precision_at_k']:.6f} MAP@100={val_m['map_at_k']:.6f} NDCG@100={val_m['ndcg_at_k']:.6f} n={val_m['n_users_eval']}",
    f"TEST P@100={test_m['precision_at_k']:.6f} MAP@100={test_m['map_at_k']:.6f} NDCG@100={test_m['ndcg_at_k']:.6f} n={test_m['n_users_eval']}",
]
(spark.createDataFrame([(l,) for l in summary], ["value"])
      .coalesce(1).write.mode("overwrite")
      .text(f"{OUT_DIR}/metrics_summary.txt"))

spark.stop()
print("[D3] Done.")

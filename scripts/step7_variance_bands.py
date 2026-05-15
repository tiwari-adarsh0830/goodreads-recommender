"""
Step 7: Bootstrap variance bands on the 1% small-split ranking metrics.

Author: at7095 and ss21040 (Goodreads Capstone, DSGA1004 Big Data, Spring 2026)

What this does
--------------
The numbers reported for D2/D4/D5 in the report are point estimates on a
single 1% user sample of val/test. A 4% absolute gap between D5 and D4
(0.0765 vs 0.0737 TEST MAP@100) is small enough that part of it could
plausibly be sampling noise. This script attaches bootstrap 95% CIs to
the metrics without retraining any model.

How
---
Per-user metrics are computed once from the saved predictions Parquet
(written by step3/step5/step6 eval scripts) joined to the held-out
ground truth. Predictions and ground truth are collected to the driver
-- the small split has ~7,600 users, so this fits comfortably -- and
B=500 bootstrap resamples are drawn with NumPy. We report mean +/- std
and the 2.5%/97.5% percentile CI for Precision@100, MAP@100, NDCG@100.

We bootstrap *over users* (not over interactions) because each user is
an independent unit of evaluation under RankingMetrics. This is the
same resampling unit Spark uses internally.

Run:
    spark-submit --master yarn --deploy-mode client \\
        --num-executors 2 --executor-cores 2 --executor-memory 4G \\
        --driver-memory 4G \\
        ~/goodreads_capstone/scripts/step7_variance_bands.py \\
        --model d5 --split test --b 500
"""
import argparse
import os
import sys
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ---- Constants ----------------------------------------------------------
NETID    = os.environ.get("NETID", "ss21040")
BASE     = f"hdfs:///user/{NETID}_nyu_edu/goodreads"
K_EVAL   = 100
RNG_SEED = 1004

MODEL_PATHS = {
    "d2": f"{BASE}/popularity",                  # POP_READ preds (see notes)
    "d4": f"{BASE}/als_implicit/full",
    "d4v2": f"{BASE}/als_implicit_v2/full",
    "d5": f"{BASE}/als_combined/full",
}

# ---- Args ---------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=list(MODEL_PATHS), required=True,
                help="which deliverable's saved predictions to bootstrap")
ap.add_argument("--split", choices=["val", "test"], default="test")
ap.add_argument("--b", type=int, default=500, help="bootstrap resamples")
args = ap.parse_args()

PRED_DIR = MODEL_PATHS[args.model]
PRED_PATH = f"{PRED_DIR}/{args.split}_small_preds"
GT_SPLIT_PATH = f"{BASE}/{args.split}_small.parquet"
OUT_DIR = f"{PRED_DIR}/variance_{args.split}"

# ---- Spark session ------------------------------------------------------
spark = (SparkSession.builder
         .appName(f"step7_variance_{args.model}_{args.split}")
         .config("spark.sql.shuffle.partitions", "100")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

print(f"=== Step 7 (variance bands) | model={args.model} split={args.split} B={args.b} ===")
print(f"preds : {PRED_PATH}")
print(f"truth : {GT_SPLIT_PATH}")

# ---- Build (user, preds, truth) collected list --------------------------
preds = spark.read.parquet(PRED_PATH).select("user_id", "recs")
split_df = spark.read.parquet(GT_SPLIT_PATH)
truth = (split_df.filter(F.col("is_read") == 1)
                 .groupBy("user_id")
                 .agg(F.collect_set("book_id").alias("ground_truth")))

joined = (preds.alias("p")
          .join(truth.alias("t"), "user_id", "inner")
          .filter((F.size("recs") > 0) & (F.size("ground_truth") > 0))
          .select("recs", "ground_truth"))

rows = joined.collect()
n_users = len(rows)
print(f"users in joined eval set : {n_users:,}")
assert n_users > 0, "Empty joined eval set; check paths."

# ---- Per-user metrics in NumPy -----------------------------------------
def per_user_metrics(rec_list, gt_list, k=K_EVAL):
    truth_set = set(gt_list)
    rec_k = rec_list[:k]
    hits = [1 if b in truth_set else 0 for b in rec_k]
    n_rel = sum(hits)

    precision = n_rel / k

    if n_rel == 0:
        ap = 0.0
    else:
        cum = 0
        ap_acc = 0.0
        denom = min(len(truth_set), k)
        for i, h in enumerate(hits, start=1):
            if h:
                cum += 1
                ap_acc += cum / i
        ap = ap_acc / denom

    dcg = 0.0
    for i, h in enumerate(hits, start=1):
        if h:
            dcg += 1.0 / np.log2(i + 1)
    ideal = min(len(truth_set), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal + 1))
    ndcg = (dcg / idcg) if idcg > 0 else 0.0

    return precision, ap, ndcg

p_arr = np.zeros(n_users, dtype=np.float64)
ap_arr = np.zeros(n_users, dtype=np.float64)
ndcg_arr = np.zeros(n_users, dtype=np.float64)

for i, r in enumerate(rows):
    p, ap, ndcg = per_user_metrics(list(r["recs"]), list(r["ground_truth"]))
    p_arr[i] = p
    ap_arr[i] = ap
    ndcg_arr[i] = ndcg

mean_p = float(p_arr.mean())
mean_ap = float(ap_arr.mean())
mean_ndcg = float(ndcg_arr.mean())
print(f"point estimate : P@100={mean_p:.4f}  MAP@100={mean_ap:.4f}  NDCG@100={mean_ndcg:.4f}")

# ---- Bootstrap ----------------------------------------------------------
rng = np.random.default_rng(RNG_SEED)
boot_p = np.empty(args.b, dtype=np.float64)
boot_ap = np.empty(args.b, dtype=np.float64)
boot_ndcg = np.empty(args.b, dtype=np.float64)

for b in range(args.b):
    idx = rng.integers(0, n_users, size=n_users)
    boot_p[b] = p_arr[idx].mean()
    boot_ap[b] = ap_arr[idx].mean()
    boot_ndcg[b] = ndcg_arr[idx].mean()

def summary(name, point, draws):
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return (f"{name:<10} point={point:.6f}  "
            f"mean={draws.mean():.6f}  sd={draws.std(ddof=1):.6f}  "
            f"95% CI=[{lo:.6f}, {hi:.6f}]")

lines = [
    f"=== Step 7 variance bands | model={args.model} split={args.split} ===",
    f"users in eval set : {n_users:,}",
    f"bootstrap B       : {args.b}",
    f"RNG seed          : {RNG_SEED}",
    "",
    summary("P@100",    mean_p,    boot_p),
    summary("MAP@100",  mean_ap,   boot_ap),
    summary("NDCG@100", mean_ndcg, boot_ndcg),
]
for l in lines:
    print(l)

(spark.createDataFrame([(l,) for l in lines], ["value"])
      .coalesce(1)
      .write.mode("overwrite")
      .text(OUT_DIR))

print(f"[step7] Written -> {OUT_DIR}")
spark.stop()

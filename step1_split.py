"""
Step 1: User-stratified 70/15/15 train/val/test split + 1% prototyping sample.

Author : at7095 (Goodreads Capstone, DSGA1004 Big Data, Spring 2026)

Design (defended in the report)
-------------------------------
* Granularity        : user-stratified (per-user holdout)
* Method             : random per-user holdout (no timestamps available)
* Train fraction     : 0.70 of each eligible user's interactions
* Val fraction       : 0.15
* Test fraction      : 0.15
* Eligibility        : users with >= MIN_INT_PER_USER (=10) interactions
                       are split; users with fewer go entirely into train.
* Reproducibility    : single seed SPLIT_SEED=1004 used in all rand() calls
* Filtering          : we do NOT filter rating=0 at this stage; the explicit
                       model in Deliverable 3 will filter when training. The
                       split is signal-agnostic.

Run:
    spark-submit --master yarn --deploy-mode client \\
        --num-executors 3 --executor-cores 4 --executor-memory 6G \\
        --driver-memory 2G \\
        --conf spark.dynamicAllocation.enabled=true \\
        --conf spark.dynamicAllocation.minExecutors=1 \\
        --conf spark.dynamicAllocation.maxExecutors=4 \\
        ~/goodreads_capstone/scripts/step1_split.py \\
        2>&1 | tee ~/goodreads_capstone/logs/step1_$(date +%Y%m%d_%H%M%S).log

HDFS outputs:
    /user/at7095_nyu_edu/goodreads/train.parquet
    /user/at7095_nyu_edu/goodreads/val.parquet
    /user/at7095_nyu_edu/goodreads/test.parquet
    /user/at7095_nyu_edu/goodreads/train_small.parquet
    /user/at7095_nyu_edu/goodreads/val_small.parquet
    /user/at7095_nyu_edu/goodreads/test_small.parquet

Local report artifacts:
    ~/goodreads_capstone/report_artifacts/step1_split_stats.txt
    ~/goodreads_capstone/report_artifacts/step1_small_split_stats.txt
"""
import os
import time
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark import StorageLevel

# ---- Constants ----------------------------------------------------------
NETID                = "at7095"
DATA_DIR             = f"hdfs:///user/{NETID}_nyu_edu/goodreads"
LOCAL_REPORT_DIR     = os.path.expanduser("~/goodreads_capstone/report_artifacts")
os.makedirs(LOCAL_REPORT_DIR, exist_ok=True)

MIN_INT_PER_USER     = 10        # eligibility threshold for val/test
TRAIN_FRAC           = 0.70
VAL_FRAC             = 0.15      # implies TEST_FRAC = 0.15
SPLIT_SEED           = 1004
SMALL_USER_SAMPLE_FR = 0.01      # 1% user sample for prototyping
SMALL_SAMPLE_SEED    = 1004

# ---- Spark session ------------------------------------------------------
spark = (SparkSession.builder
    .appName(f"{NETID}_step1_split")
    .config("spark.sql.shuffle.partitions", "200")
    .config("spark.sql.parquet.compression.codec", "snappy")
    .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

t0 = time.time()

# ---- Load Parquet from Step 0 -------------------------------------------
interactions = spark.read.parquet(f"{DATA_DIR}/interactions.parquet")
n_total = interactions.count()
print(f"Loaded interactions: {n_total:,} rows")

# ---- Tag eligibility ----------------------------------------------------
# user_total = count of interactions per user (window aggregation)
# eligible   = user_total >= MIN_INT_PER_USER
user_window = Window.partitionBy("user_id")
tagged = (interactions
    .withColumn("user_total", F.count("*").over(user_window))
    .withColumn("eligible", F.col("user_total") >= F.lit(MIN_INT_PER_USER)))

# ---- Per-user random shuffle + bucketing --------------------------------
# Window: partition by user, order by rand(seed). row_number/user_total -> (0,1]
shuffle_window = Window.partitionBy("user_id").orderBy(F.rand(seed=SPLIT_SEED))
bucketed = (tagged
    .withColumn("rn", F.row_number().over(shuffle_window))
    .withColumn("frac", F.col("rn") / F.col("user_total"))
    # Eligible users: split by frac. Non-eligible users: all to train.
    .withColumn("split",
        F.when(~F.col("eligible"), F.lit("train"))
         .when(F.col("frac") <= F.lit(TRAIN_FRAC), F.lit("train"))
         .when(F.col("frac") <= F.lit(TRAIN_FRAC + VAL_FRAC), F.lit("val"))
         .otherwise(F.lit("test"))))

# We persist because we'll filter it 3x (one per split) and run several
# verification queries.
bucketed = bucketed.persist(StorageLevel.MEMORY_AND_DISK)

# ---- Carve out splits, stripped of helper columns -----------------------
keep_cols = ["user_id", "book_id", "is_read", "rating", "is_reviewed"]
train = bucketed.filter(F.col("split") == "train").select(*keep_cols)
val   = bucketed.filter(F.col("split") == "val"  ).select(*keep_cols)
test  = bucketed.filter(F.col("split") == "test" ).select(*keep_cols)

# ---- Split stats --------------------------------------------------------
n_train = train.count()
n_val   = val.count()
n_test  = test.count()

eligible_users    = bucketed.filter(F.col("eligible")).select("user_id").distinct().count()
ineligible_users  = bucketed.filter(~F.col("eligible")).select("user_id").distinct().count()

users_in_train = train.select("user_id").distinct().count()
users_in_val   = val.select("user_id").distinct().count()
users_in_test  = test.select("user_id").distinct().count()

split_stats = (
    f"=== Split statistics (seed={SPLIT_SEED}, min_int={MIN_INT_PER_USER}, "
    f"frac={TRAIN_FRAC}/{VAL_FRAC}/{1-TRAIN_FRAC-VAL_FRAC:.2f}) ===\n"
    f"total interactions  : {n_total:>15,}\n"
    f"train rows          : {n_train:>15,}  ({n_train/n_total*100:5.2f}%)\n"
    f"val   rows          : {n_val:>15,}  ({n_val/n_total*100:5.2f}%)\n"
    f"test  rows          : {n_test:>15,}  ({n_test/n_total*100:5.2f}%)\n"
    f"\n"
    f"eligible users (>= {MIN_INT_PER_USER} ints)  : {eligible_users:>10,}\n"
    f"ineligible users (< {MIN_INT_PER_USER} ints) : {ineligible_users:>10,}\n"
    f"users appearing in train      : {users_in_train:>10,}\n"
    f"users appearing in val        : {users_in_val:>10,}\n"
    f"users appearing in test       : {users_in_test:>10,}\n"
)
print(split_stats)

# ---- INVARIANT CHECKS ---------------------------------------------------
# These guard against silent split bugs that would invalidate downstream
# evaluation. Failure halts the job so we can fix it before training models.

# Invariant 1: no row loss
assert n_train + n_val + n_test == n_total, \
    f"Row sum mismatch: {n_train + n_val + n_test:,} != {n_total:,}"
print("[OK] Invariant 1: train+val+test sum equals total")

# Invariant 2: every val/test user appears in train
val_users  = val.select("user_id").distinct()
test_users = test.select("user_id").distinct()
train_users_df = train.select("user_id").distinct()
val_orphans  = val_users.join(train_users_df, "user_id", "left_anti").count()
test_orphans = test_users.join(train_users_df, "user_id", "left_anti").count()
assert val_orphans  == 0, f"{val_orphans} val users have no training data"
assert test_orphans == 0, f"{test_orphans} test users have no training data"
print("[OK] Invariant 2: every val/test user has training data (no cold-start in eval)")

# Invariant 3: val and test users are exactly the eligible users
# (Within tiny tolerance: a user with exactly MIN_INT_PER_USER interactions
#  splits into 7-train/2-val/1-test typically. They WILL all show up in val
#  and test as long as ceil(MIN_INT_PER_USER * 0.15) >= 1.)
assert users_in_val  <= eligible_users, "More val users than eligible users?"
assert users_in_test <= eligible_users, "More test users than eligible users?"
val_coverage  = users_in_val  / eligible_users
test_coverage = users_in_test / eligible_users
print(f"[INFO] val users / eligible  = {val_coverage*100:5.2f}%")
print(f"[INFO] test users / eligible = {test_coverage*100:5.2f}%")
assert val_coverage  >= 0.99, f"Only {val_coverage*100:.2f}% of eligible users got val rows"
assert test_coverage >= 0.99, f"Only {test_coverage*100:.2f}% of eligible users got test rows"
print("[OK] Invariant 3: >=99% of eligible users have val and test rows")

# Invariant 4: no row appears in two splits
# Compute via bucketed (the helper DF) for efficiency; each row has exactly
# one 'split' value so a duplicate would be a logical impossibility, but
# we verify anyway via union+count.
union_three = train.unionAll(val).unionAll(test)
n_union = union_three.count()
assert n_union == n_total, f"Union has {n_union:,} rows, expected {n_total:,}"
print("[OK] Invariant 4: train+val+test rows are disjoint")

with open(os.path.join(LOCAL_REPORT_DIR, "step1_split_stats.txt"), "w") as f:
    f.write(split_stats)

# ---- Write full splits --------------------------------------------------
print("Writing train.parquet ...")
train.repartition(150).write.mode("overwrite").parquet(f"{DATA_DIR}/train.parquet")
print("Writing val.parquet ...")
val.repartition(40).write.mode("overwrite").parquet(f"{DATA_DIR}/val.parquet")
print("Writing test.parquet ...")
test.repartition(40).write.mode("overwrite").parquet(f"{DATA_DIR}/test.parquet")

# ---- Build small (1% user) prototyping splits ---------------------------
# Rationale: brief recommends user-based subsampling for prototyping. We
# sample 1% of users at random and keep ALL their interactions, then apply
# the same split logic on the subsample (using the SAME seed so a user's
# split assignment is consistent). This gives us a fast iteration loop for
# ALS hyperparameter tuning without ever pretending it's the real eval.
print("\n--- Building 1%-user prototyping splits ---")
all_users = bucketed.select("user_id").distinct()
sample_users = all_users.sample(False, SMALL_USER_SAMPLE_FR, seed=SMALL_SAMPLE_SEED)
n_sample_users = sample_users.count()
print(f"Sampled users: {n_sample_users:,}")

bucketed_small = bucketed.join(F.broadcast(sample_users), "user_id", "inner")

train_s = bucketed_small.filter(F.col("split") == "train").select(*keep_cols)
val_s   = bucketed_small.filter(F.col("split") == "val"  ).select(*keep_cols)
test_s  = bucketed_small.filter(F.col("split") == "test" ).select(*keep_cols)

n_train_s = train_s.count()
n_val_s   = val_s.count()
n_test_s  = test_s.count()
small_stats = (
    f"=== Small (1% user) prototyping splits ===\n"
    f"sampled users : {n_sample_users:,}\n"
    f"train rows    : {n_train_s:,}\n"
    f"val   rows    : {n_val_s:,}\n"
    f"test  rows    : {n_test_s:,}\n"
)
print(small_stats)
with open(os.path.join(LOCAL_REPORT_DIR, "step1_small_split_stats.txt"), "w") as f:
    f.write(small_stats)

print("Writing train_small.parquet ...")
train_s.repartition(8).write.mode("overwrite").parquet(f"{DATA_DIR}/train_small.parquet")
print("Writing val_small.parquet ...")
val_s.repartition(4).write.mode("overwrite").parquet(f"{DATA_DIR}/val_small.parquet")
print("Writing test_small.parquet ...")
test_s.repartition(4).write.mode("overwrite").parquet(f"{DATA_DIR}/test_small.parquet")

bucketed.unpersist()
elapsed = time.time() - t0
print(f"\n[Step 1 complete in {elapsed/60:.1f} min]")
spark.stop()

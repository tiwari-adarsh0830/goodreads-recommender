"""
Step 0: CSV -> Parquet conversion + distributional sanity statistics.

Author : at7095 (Goodreads Capstone, DSGA1004 Big Data, Spring 2026)

Purpose
-------
One-shot preprocessing job. Reads the three raw CSVs from the shared HDFS
location, casts to explicit schemas (no inferSchema on 223M rows), writes
Parquet with snappy compression, and emits summary statistics used later to
justify (a) the rating>0 filter for the explicit-feedback model and (b) the
minimum-interactions threshold for train/val/test splitting.

Run command (client mode keeps driver on the login node so we can see logs;
driver memory stays under the documented 3 GB login-node cap):

    spark-submit --master yarn --deploy-mode client \\
        --num-executors 3 --executor-cores 4 --executor-memory 6G \\
        --driver-memory 2G \\
        --conf spark.dynamicAllocation.enabled=true \\
        --conf spark.dynamicAllocation.minExecutors=1 \\
        --conf spark.dynamicAllocation.maxExecutors=4 \\
        ~/goodreads_capstone/scripts/step0_csv_to_parquet.py \\
        2>&1 | tee ~/goodreads_capstone/logs/step0_$(date +%Y%m%d_%H%M%S).log

Outputs (HDFS)
--------------
    /user/at7095_nyu_edu/goodreads/interactions.parquet
    /user/at7095_nyu_edu/goodreads/user_id_map.parquet
    /user/at7095_nyu_edu/goodreads/book_id_map.parquet

Outputs (local, for the report)
-------------------------------
    ~/goodreads_capstone/report_artifacts/step0_counts.txt
    ~/goodreads_capstone/report_artifacts/step0_rating_dist.csv
    ~/goodreads_capstone/report_artifacts/step0_isread_dist.csv
    ~/goodreads_capstone/report_artifacts/step0_isreviewed_dist.csv
    ~/goodreads_capstone/report_artifacts/step0_user_activity_percentiles.csv
"""
import os
import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, IntegerType, StringType, ByteType
)
from pyspark import StorageLevel

# ---- Constants ----------------------------------------------------------
NETID   = "at7095"
SRC_DIR = "hdfs:///user/pw44_nyu_edu"
DST_DIR = f"hdfs:///user/{NETID}_nyu_edu/goodreads"
LOCAL_REPORT_DIR = os.path.expanduser("~/goodreads_capstone/report_artifacts")
os.makedirs(LOCAL_REPORT_DIR, exist_ok=True)

# ---- Spark session ------------------------------------------------------
# AQE left at default (on). It dynamically coalesces shuffle partitions and
# is a strict win for analytical jobs like the groupBy stats below.
spark = (SparkSession.builder
    .appName(f"{NETID}_step0_csv_to_parquet")
    .config("spark.sql.shuffle.partitions", "200")
    .config("spark.sql.parquet.compression.codec", "snappy")
    .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

# ---- Explicit schemas ---------------------------------------------------
# Why explicit:
#   1. inferSchema requires a full extra pass over a 4.3 GB file.
#   2. inferSchema can silently mis-type columns (e.g. read rating as string
#      if the first batch contains nulls).
#   3. Using ByteType for is_read / is_reviewed / rating cuts storage by ~75%
#      vs IntegerType in Parquet (column-wise dictionary + RLE encoding).
#   We confirmed the column ordering by inspecting the CSV header in Step 0.2.
interactions_schema = StructType([
    StructField("user_id",     IntegerType(), nullable=False),
    StructField("book_id",     IntegerType(), nullable=False),
    StructField("is_read",     ByteType(),    nullable=True),
    StructField("rating",      ByteType(),    nullable=True),
    StructField("is_reviewed", ByteType(),    nullable=True),
])
user_map_schema = StructType([
    StructField("user_id_csv", IntegerType(), nullable=False),
    StructField("user_id",     StringType(),  nullable=True),  # original Goodreads hash
])
book_map_schema = StructType([
    StructField("book_id_csv", IntegerType(), nullable=False),
    StructField("book_id",     IntegerType(), nullable=True),  # original Goodreads book id
])

# ---- Read CSVs ----------------------------------------------------------
# mode=FAILFAST: surfaces malformed rows immediately. If the job dies here
# we will switch to PERMISSIVE and inspect the rejected rows; for a clean
# distribution like Goodreads this should never trigger.
t0 = time.time()
interactions_raw = (spark.read
    .option("header", "true")
    .option("mode", "FAILFAST")
    .schema(interactions_schema)
    .csv(f"{SRC_DIR}/goodreads_interactions.csv"))

user_map = (spark.read
    .option("header", "true").option("mode", "FAILFAST")
    .schema(user_map_schema)
    .csv(f"{SRC_DIR}/user_id_map.csv"))

book_map = (spark.read
    .option("header", "true").option("mode", "FAILFAST")
    .schema(book_map_schema)
    .csv(f"{SRC_DIR}/book_id_map.csv"))

# We persist the interactions DataFrame at MEMORY_AND_DISK before computing
# multiple stats. Without this Spark would re-read the 4.3 GB CSV once per
# count/groupBy. With it, we read the CSV ONCE.
# Storage level note: pure MEMORY_ONLY is risky (eviction -> recompute);
# DISK_ONLY would defeat the purpose. MEMORY_AND_DISK_SER is the safest
# choice on this small cluster.
interactions = interactions_raw.persist(StorageLevel.MEMORY_AND_DISK)

# ---- Sanity counts ------------------------------------------------------
n_interactions   = interactions.count()           # materializes the cache
n_users_distinct = interactions.select("user_id").distinct().count()
n_books_distinct = interactions.select("book_id").distinct().count()
n_users_map      = user_map.count()
n_books_map      = book_map.count()

counts_text = (
    f"=== Goodreads dataset sanity counts ({NETID} step0) ===\n"
    f"interactions rows                : {n_interactions:>15,}\n"
    f"distinct users (in interactions) : {n_users_distinct:>15,}\n"
    f"distinct books (in interactions) : {n_books_distinct:>15,}\n"
    f"user_id_map rows                 : {n_users_map:>15,}\n"
    f"book_id_map rows                 : {n_books_map:>15,}\n"
    f"\nBrief expected ~223M interactions, ~876K users, ~2.4M books.\n"
)
print(counts_text)
with open(os.path.join(LOCAL_REPORT_DIR, "step0_counts.txt"), "w") as f:
    f.write(counts_text)

# ---- Distributional stats (small enough to bring to driver) -------------
# Rating distribution: critical for justifying rating>0 filter (rating=0
# means "not rated" per Goodreads convention -- it is *not* a 0-star rating).
print("Computing rating distribution...")
rating_dist = (interactions.groupBy("rating").count().orderBy("rating")
               .toPandas())
rating_dist.to_csv(os.path.join(LOCAL_REPORT_DIR, "step0_rating_dist.csv"),
                   index=False)
print(rating_dist.to_string(index=False))

print("\nComputing is_read distribution...")
isread_dist = (interactions.groupBy("is_read").count().orderBy("is_read")
               .toPandas())
isread_dist.to_csv(os.path.join(LOCAL_REPORT_DIR, "step0_isread_dist.csv"),
                   index=False)
print(isread_dist.to_string(index=False))

print("\nComputing is_reviewed distribution...")
isrev_dist = (interactions.groupBy("is_reviewed").count().orderBy("is_reviewed")
              .toPandas())
isrev_dist.to_csv(os.path.join(LOCAL_REPORT_DIR, "step0_isreviewed_dist.csv"),
                  index=False)
print(isrev_dist.to_string(index=False))

# Per-user activity percentiles: needed to choose the min-interactions
# threshold for splitting (Step 1 of Deliverable methodology).
# approx_percentile uses GK-sketch; tolerance is fine for thresholding.
print("\nComputing per-user activity percentiles...")
per_user_stats = (interactions
    .groupBy("user_id").agg(F.count("*").alias("n"))
    .selectExpr(
        "min(n) as min_int",
        "approx_percentile(n, 0.10) as p10",
        "approx_percentile(n, 0.25) as p25",
        "approx_percentile(n, 0.50) as p50",
        "approx_percentile(n, 0.75) as p75",
        "approx_percentile(n, 0.90) as p90",
        "approx_percentile(n, 0.95) as p95",
        "approx_percentile(n, 0.99) as p99",
        "max(n) as max_int",
        "avg(n) as mean_int",
    ).toPandas())
per_user_stats.to_csv(
    os.path.join(LOCAL_REPORT_DIR, "step0_user_activity_percentiles.csv"),
    index=False)
print(per_user_stats.to_string(index=False))

# ---- Write Parquet ------------------------------------------------------
# repartition(200) before write -> ~200 output files, each ~5-15 MB after
# Snappy compression. This is the Spark/HDFS sweet spot:
#   - Too few (< ~50): poor parallelism downstream.
#   - Too many (> ~1000): NameNode pressure, small-file problem.
# We do NOT partitionBy(any_column): partitionBy(user_id) would create
# ~876K HDFS directories (catastrophic). partitionBy(rating) would skew
# heavily (most rows have rating=0). Plain Parquet with column statistics
# is the right call here; downstream filters on rating/is_read still
# benefit from row-group min/max pruning.
print("\nWriting interactions.parquet ...")
(interactions
   .repartition(200)
   .write.mode("overwrite")
   .parquet(f"{DST_DIR}/interactions.parquet"))

print("Writing user_id_map.parquet ...")
(user_map.coalesce(4)
   .write.mode("overwrite")
   .parquet(f"{DST_DIR}/user_id_map.parquet"))

print("Writing book_id_map.parquet ...")
(book_map.coalesce(4)
   .write.mode("overwrite")
   .parquet(f"{DST_DIR}/book_id_map.parquet"))

# Now safe to release the cache.
interactions.unpersist()

# ---- Round-trip verification --------------------------------------------
# Read back from Parquet and confirm row count matches. If this fails,
# we abort -- nothing downstream is trustworthy.
roundtrip = spark.read.parquet(f"{DST_DIR}/interactions.parquet")
n_after = roundtrip.count()
assert n_after == n_interactions, \
    f"FATAL: row count mismatch! parquet={n_after:,} csv={n_interactions:,}"
print(f"\n[OK] Parquet round-trip verified: {n_after:,} rows")

elapsed = time.time() - t0
print(f"\n[Step 0 complete in {elapsed/60:.1f} min]")

spark.stop()

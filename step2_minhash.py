"""
Step 2 (Deliverable 1) v3: MinHash + LSH market segmentation.

Author: at7095 and ss21040 (Goodreads Capstone, DSGA1004 Big Data, Spring 2026)

Why MIN_READERS=200
-------------------
v1 (MIN_READERS=100) and v2 (MIN_READERS=100, slim DF) both hit Spark's
driver maxResultSize cap because the candidate-pair set produced by
approxSimilarityJoin was multi-GB even when projected to (book_a, book_b,
distance). The dominant cost is the candidate-set fan-out, which scales
roughly as n_books * avg_collisions_per_book.

Raising MIN_READERS to 200 cuts n_books from ~125K to ~60K and the
candidate set by a factor of >2, which together with maxResultSize=0
keeps the join tractable on the 3-worker cluster. The substantive answer
is unchanged: books with <200 readers are not "market positions" in any
meaningful sense -- their Jaccard similarities are dominated by sampling
noise from small denominators.

Pipeline (unchanged from v2 except as noted)
-------------------------------------------
1. Load full interactions.parquet, filter is_read=1.
2. Per-book reader sets; keep books with >= 200 readers.
3. Build sparse binary user vectors (dimension = NUM_USERS, sorted indices).
4. Project to a SLIM df (book_id + features only) and fit MinHashLSH(K=32).
5. approxSimilarityJoin self-join with Jaccard-distance threshold = 0.3
   (sim >= 0.7), output ONLY (book_a, book_b, distance) -- no payload.
6. Filter book_a < book_b. NO intermediate Parquet checkpoint (v2 step
   that failed); we let Spark stream the candidates directly into the
   downstream join.
7. Re-join book_users on book_a and book_b to get users_a / users_b
   only for surviving candidates.
8. Compute exact Jaccard via array_intersect / array_union.
9. Top 100 by exact Jaccard descending.
10. Join book_id_map for original Goodreads IDs.
11. Write directly to HDFS as single-file CSV + Parquet.

Why MinHash + LSH (general motivation)
--------------------------------------
* Brute-force all-pairs Jaccard over 60K books is C(60K, 2) ~= 1.8e9 pairs:
  infeasible for a 3-worker cluster.
* MinHash signatures collide with probability = Jaccard similarity (course
  bigD10 slides). Spark's MinHashLSH uses single-row bands: each of K hash
  functions is its own bucket; collision in any flags a candidate.
* For sim >= 0.7 (our LSH threshold) and K=32, recall is essentially 1
  (1 - (1-0.7)^32 ~= 1 - 5e-17). Final reported similarity is computed
  *exactly* via array_intersect/array_union on the much smaller candidate
  set, so K affects recall, not the precision of final Jaccard numbers.

Run command (cluster mode):
    spark-submit --master yarn --deploy-mode cluster \\
        --num-executors 3 --executor-cores 4 --executor-memory 6G \\
        --driver-cores 2 --driver-memory 6G \\
        --conf spark.dynamicAllocation.enabled=true \\
        --conf spark.dynamicAllocation.minExecutors=1 \\
        --conf spark.dynamicAllocation.maxExecutors=4 \\
        --conf spark.sql.shuffle.partitions=600 \\
        --conf spark.network.timeout=600s \\
        --conf spark.executor.heartbeatInterval=60s \\
        --conf spark.driver.maxResultSize=0 \\
        --conf spark.yarn.maxAppAttempts=1 \\
        ~/goodreads_capstone/scripts/step2_minhash.py
"""
import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import MinHashLSH
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark import StorageLevel

# ---- Constants ----------------------------------------------------------
NETID            = "at7095"
DATA_DIR         = f"hdfs:///user/{NETID}_nyu_edu/goodreads"
OUT_DIR          = f"{DATA_DIR}/minhash"

MIN_READERS      = 200   # popularity floor for "meaningful market position"
NUM_HASH_TABLES  = 32
LSH_THRESHOLD    = 0.3   # Jaccard distance -> sim >= 0.7
TOP_K            = 100
SEED             = 1004

# ---- Spark session ------------------------------------------------------
spark = (SparkSession.builder
    .appName(f"{NETID}_step2_minhash_v3")
    .config("spark.sql.parquet.compression.codec", "snappy")
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

t0 = time.time()

# ---- 2.1: Load and filter -----------------------------------------------
print("Loading interactions...")
inter = spark.read.parquet(f"{DATA_DIR}/interactions.parquet")
read_inter = inter.filter(F.col("is_read") == 1).select("user_id", "book_id")
n_read = read_inter.count()
print(f"is_read=1 interactions: {n_read:,}")

NUM_USERS = (read_inter.agg(F.max("user_id")).collect()[0][0]) + 1
print(f"NUM_USERS (vector dimension): {NUM_USERS:,}")

# ---- 2.2: Per-book reader sets, popularity filter -----------------------
print(f"Building per-book reader sets (MIN_READERS={MIN_READERS})...")
book_users = (read_inter
    .groupBy("book_id")
    .agg(F.array_sort(F.collect_set("user_id")).alias("user_ids"))
    .withColumn("n_readers", F.size("user_ids"))
    .filter(F.col("n_readers") >= F.lit(MIN_READERS)))
book_users = book_users.persist(StorageLevel.MEMORY_AND_DISK)
n_books = book_users.count()
print(f"Books surviving popularity filter: {n_books:,}")
assert n_books > 1000, f"Only {n_books} books survived; check MIN_READERS"

# ---- 2.3: Build sparse binary vectors -----------------------------------
def to_sparse_vector(user_ids, dim=NUM_USERS):
    return Vectors.sparse(dim, user_ids, [1.0] * len(user_ids))
to_sparse_udf = F.udf(to_sparse_vector, VectorUDT())

book_vectors = (book_users
    .withColumn("features", to_sparse_udf(F.col("user_ids")))
    .select("book_id", "features"))    # SLIM: no user_ids passed to LSH
book_vectors = book_vectors.persist(StorageLevel.MEMORY_AND_DISK)
_ = book_vectors.count()
print("Vectors materialised.")

# ---- 2.4: Fit MinHashLSH ------------------------------------------------
print(f"Fitting MinHashLSH (numHashTables={NUM_HASH_TABLES})...")
mh = MinHashLSH(inputCol="features", outputCol="hashes",
                numHashTables=NUM_HASH_TABLES, seed=SEED)
mh_model = mh.fit(book_vectors)
hashed = mh_model.transform(book_vectors).persist(StorageLevel.MEMORY_AND_DISK)
_ = hashed.count()
print("MinHash signatures computed.")

# ---- 2.5: approxSimilarityJoin, project to IDs only ---------------------
# v3 does NOT checkpoint the candidate set to Parquet (v2's step that
# failed). Instead the candidates DF is consumed directly by the downstream
# join + Jaccard computation. Spark's catalyst optimizer fuses the join
# with the array_intersect/array_union step, so the heavy user_ids payload
# never lands at any single coordinator point.
print(f"approxSimilarityJoin (threshold={LSH_THRESHOLD})...")
joined = mh_model.approxSimilarityJoin(
    hashed, hashed, threshold=LSH_THRESHOLD, distCol="JaccardDistance")

candidates = (joined
    .select(
        F.col("datasetA.book_id").alias("book_a"),
        F.col("datasetB.book_id").alias("book_b"),
        F.col("JaccardDistance").alias("mh_distance"))
    .filter(F.col("book_a") < F.col("book_b"))
    .withColumn("mh_similarity", F.lit(1.0) - F.col("mh_distance")))

# ---- 2.6: Re-attach user_ids to candidates -----------------------------
ba = book_users.select(F.col("book_id").alias("book_a"),
                       F.col("user_ids").alias("users_a"))
bb = book_users.select(F.col("book_id").alias("book_b"),
                       F.col("user_ids").alias("users_b"))
candidates_full = (candidates
    .join(ba, "book_a", "inner")
    .join(bb, "book_b", "inner"))

# ---- 2.7: Exact Jaccard -------------------------------------------------
print("Computing exact Jaccard for candidate pairs...")
exact = (candidates_full
    .withColumn("intersect_size",
                F.size(F.array_intersect(F.col("users_a"), F.col("users_b"))))
    .withColumn("union_size",
                F.size(F.array_union(F.col("users_a"), F.col("users_b"))))
    .withColumn("exact_jaccard",
                F.col("intersect_size") / F.col("union_size"))
    .select("book_a", "book_b", "mh_similarity", "exact_jaccard",
            "intersect_size", "union_size"))

# ---- 2.8: Top 100 by exact Jaccard --------------------------------------
top = (exact
    .orderBy(F.col("exact_jaccard").desc())
    .limit(TOP_K))
top = top.persist(StorageLevel.MEMORY_AND_DISK)
n_top = top.count()
print(f"Top-{TOP_K} computed (n={n_top}).")
assert n_top == TOP_K, f"Expected {TOP_K} pairs, got {n_top}"

hashed.unpersist()
book_vectors.unpersist()

# ---- 2.9: Surface original Goodreads IDs --------------------------------
book_map = spark.read.parquet(f"{DATA_DIR}/book_id_map.parquet")
top_named = (top
    .join(book_map.withColumnRenamed("book_id_csv", "book_a")
                  .withColumnRenamed("book_id", "goodreads_id_a"),
          on="book_a", how="left")
    .join(book_map.withColumnRenamed("book_id_csv", "book_b")
                  .withColumnRenamed("book_id", "goodreads_id_b"),
          on="book_b", how="left")
    .select(
        "book_a", "goodreads_id_a",
        "book_b", "goodreads_id_b",
        "exact_jaccard", "mh_similarity",
        "intersect_size", "union_size")
    .orderBy(F.col("exact_jaccard").desc()))

# ---- 2.10: Write to HDFS ------------------------------------------------
print("Writing top-100 to HDFS (Parquet + single-CSV)...")
(top_named.coalesce(1)
   .write.mode("overwrite")
   .parquet(f"{OUT_DIR}/top100_pairs.parquet"))
(top_named.coalesce(1)
   .write.mode("overwrite")
   .option("header", "true")
   .csv(f"{OUT_DIR}/top100_pairs_csv"))

pdf = top_named.toPandas()
print(f"Top-100 written; min Jaccard = {pdf['exact_jaccard'].iloc[-1]:.4f}, "
      f"max Jaccard = {pdf['exact_jaccard'].iloc[0]:.4f}")

summary_lines = [
    f"=== Step 2 (MinHash market segmentation) v3 summary ===",
    f"is_read=1 interactions used     : {n_read:,}",
    f"NUM_USERS (vector dimension)    : {NUM_USERS:,}",
    f"MIN_READERS popularity filter   : {MIN_READERS}",
    f"books surviving filter          : {n_books:,}",
    f"NUM_HASH_TABLES                 : {NUM_HASH_TABLES}",
    f"LSH Jaccard-distance threshold  : {LSH_THRESHOLD}",
    f"top-K returned                  : {TOP_K}",
    f"top exact Jaccard               : {pdf['exact_jaccard'].iloc[0]:.4f}",
    f"100th exact Jaccard             : {pdf['exact_jaccard'].iloc[-1]:.4f}",
]
summary_df = spark.createDataFrame([(line,) for line in summary_lines], ["line"])
(summary_df.coalesce(1)
   .write.mode("overwrite")
   .text(f"{OUT_DIR}/summary_txt"))
for line in summary_lines:
    print(line)

elapsed = time.time() - t0
print(f"\n[Step 2 complete in {elapsed/60:.1f} min]")

book_users.unpersist()
top.unpersist()
spark.stop()

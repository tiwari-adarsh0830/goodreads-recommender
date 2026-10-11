"""
Step 2b (Deliverable 1 follow-up): edition-level deduplication of the top-K
similar book pairs.

Author: Adarsh Tiwari and Sheetal Sood (Goodreads Capstone, DSGA1004 Big Data, Spring 2026)

Context
-------
The Step 2 top-100 is dominated by adjacent book IDs (e.g. 459806/459807)
that turn out to be different editions of the same Goodreads work. The
MinHash output is correct -- those readers really do co-occur -- but
"different editions of Twilight" isn't a useful market-segmentation
finding. This script collapses near-duplicate book IDs into "work
groups" using union-find on the high-Jaccard candidate set, picks one
canonical book per group, and re-extracts the top-100 pairs across
distinct groups.

What counts as the same work
----------------------------
A pair (a, b) is considered same-work iff:
  exact_jaccard >= DEDUP_THRESHOLD          (default 0.80)
AND |goodreads_id_a - goodreads_id_b| <= ID_DISTANCE_MAX  (default 50)

The ID-distance gate is what filters out genuine "books with overlapping
audiences" (which can have Jaccard >= 0.80 with very different IDs) from
"editions of the same title" (which sit adjacent in the Goodreads ID
space because Goodreads issues sequential IDs for added editions).

Inputs
------
HDFS Parquet from step2_minhash.py:
    {DATA_DIR}/minhash/top100_pairs.parquet     (book_a, book_b, jaccard, ids)
    {DATA_DIR}/minhash/candidates.parquet       (optional: full candidate set)

If the full candidate Parquet is not available, this script falls back to
the top-K written by step2, which is enough to dedupe within those K
pairs but cannot promote new work-pairs into the top-100. Re-running
step2 with checkpoint=True for the candidate set is the right way to do
the full version of this analysis.

Run:
    spark-submit --master yarn --deploy-mode client \\
        --num-executors 2 --executor-cores 2 --executor-memory 4G \\
        --driver-memory 2G \\
        ~/goodreads_capstone/scripts/step2_dedup_editions.py
"""
import os
import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ---- Constants ----------------------------------------------------------
NETID            = os.environ.get("NETID", "your_netid")
DATA_DIR         = f"hdfs:///user/{NETID}_nyu_edu/goodreads"
TOP_IN           = f"{DATA_DIR}/minhash/top100_pairs.parquet"
CAND_IN          = f"{DATA_DIR}/minhash/candidates.parquet"   # optional
OUT_DIR          = f"{DATA_DIR}/minhash/deduped"
TOP_K            = 100

DEDUP_THRESHOLD  = 0.80
ID_DISTANCE_MAX  = 50

# ---- Spark session ------------------------------------------------------
spark = (SparkSession.builder
    .appName(f"{NETID}_step2_dedup")
    .config("spark.sql.parquet.compression.codec", "snappy")
    .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

t0 = time.time()

# ---- Load pairs ---------------------------------------------------------
# Prefer the full candidate set if it was checkpointed; otherwise work
# off the top-100 only and emit a warning so the report can note it.
fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
cand_exists = fs.exists(spark._jvm.org.apache.hadoop.fs.Path(CAND_IN))

if cand_exists:
    print(f"[dedup] Using full candidate set from {CAND_IN}")
    pairs = spark.read.parquet(CAND_IN).select(
        "book_a", "book_b", "exact_jaccard",
        "goodreads_id_a", "goodreads_id_b")
else:
    print(f"[dedup] WARNING: {CAND_IN} not found; falling back to top-100. "
          "Edition collapses can shrink the list below 100 -- rerun step2 "
          "with the candidate checkpoint enabled for the full analysis.")
    pairs = spark.read.parquet(TOP_IN).select(
        "book_a", "book_b", "exact_jaccard",
        "goodreads_id_a", "goodreads_id_b")

pairs = pairs.cache()
n_pairs_in = pairs.count()
print(f"[dedup] input pairs : {n_pairs_in:,}")

# ---- Edition edges ------------------------------------------------------
# A pair is an "edition edge" if it satisfies BOTH the high Jaccard and
# the small goodreads-ID gap. Goodreads issues sequential IDs for added
# editions of the same work, so the ID-gap gate is what separates
# "editions" from "books with similar audiences".
edition_edges = (pairs
    .filter(F.col("exact_jaccard") >= F.lit(DEDUP_THRESHOLD))
    .filter(F.abs(F.col("goodreads_id_a") - F.col("goodreads_id_b"))
            <= F.lit(ID_DISTANCE_MAX))
    .select("book_a", "book_b"))

n_edges = edition_edges.count()
print(f"[dedup] edition edges (jacc>={DEDUP_THRESHOLD}, |id_gap|<={ID_DISTANCE_MAX}) : {n_edges:,}")

# ---- Union-find on the driver ------------------------------------------
# Edition-edge set is small (a few thousand at most); doing union-find on
# the driver is simpler than a Pregel/GraphFrames CC job and avoids
# pulling GraphFrames into the dependency set.
edges = [(r["book_a"], r["book_b"]) for r in edition_edges.collect()]

parent = {}
def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]   # path compression
        x = parent[x]
    return x

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        # canonical = smaller book_id (stable across reruns)
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

for a, b in edges:
    union(a, b)

# every book mentioned in the input pairs needs a canonical entry, even
# isolated singletons
def all_books_iter():
    for r in pairs.select("book_a").union(pairs.select("book_b")).distinct().collect():
        yield r[0]

for b in all_books_iter():
    find(b)

canonical = {b: find(b) for b in parent}
n_groups  = len(set(canonical.values()))
print(f"[dedup] union-find: {len(canonical):,} books -> {n_groups:,} work groups")

# ---- Map books to canonical and re-rank ---------------------------------
canon_rows = [(b, c) for b, c in canonical.items()]
canon_df = spark.createDataFrame(canon_rows, ["book_id", "canonical_id"])

ca = canon_df.withColumnRenamed("book_id", "book_a") \
             .withColumnRenamed("canonical_id", "canon_a")
cb = canon_df.withColumnRenamed("book_id", "book_b") \
             .withColumnRenamed("canonical_id", "canon_b")

mapped = (pairs
    .join(ca, "book_a", "left")
    .join(cb, "book_b", "left")
    .filter(F.col("canon_a") != F.col("canon_b"))   # drop within-work pairs
    .withColumn("pair_a", F.least("canon_a", "canon_b"))
    .withColumn("pair_b", F.greatest("canon_a", "canon_b")))

# Keep the best Jaccard observed for each work-pair (any edition pairing).
best_per_workpair = (mapped
    .groupBy("pair_a", "pair_b")
    .agg(F.max("exact_jaccard").alias("exact_jaccard")))

dedup_top = (best_per_workpair
    .orderBy(F.col("exact_jaccard").desc())
    .limit(TOP_K))
n_top = dedup_top.count()
print(f"[dedup] deduped top-K rows : {n_top:,}")

# ---- Attach Goodreads IDs ----------------------------------------------
book_map = spark.read.parquet(f"{DATA_DIR}/book_id_map.parquet")
named = (dedup_top
    .join(book_map.withColumnRenamed("book_id_csv", "pair_a")
                  .withColumnRenamed("book_id", "goodreads_id_a"),
          "pair_a", "left")
    .join(book_map.withColumnRenamed("book_id_csv", "pair_b")
                  .withColumnRenamed("book_id", "goodreads_id_b"),
          "pair_b", "left")
    .select("pair_a", "goodreads_id_a",
            "pair_b", "goodreads_id_b",
            "exact_jaccard")
    .orderBy(F.col("exact_jaccard").desc()))

# ---- Write outputs ------------------------------------------------------
(named.coalesce(1)
      .write.mode("overwrite")
      .parquet(f"{OUT_DIR}/top100_pairs_deduped.parquet"))
(named.coalesce(1)
      .write.mode("overwrite")
      .option("header", "true")
      .csv(f"{OUT_DIR}/top100_pairs_deduped_csv"))

pdf = named.toPandas()
summary = [
    "=== Step 2b (edition-deduplicated top-K) ===",
    f"input pairs                 : {n_pairs_in:,}",
    f"edition edges (jacc>={DEDUP_THRESHOLD}, |id|<={ID_DISTANCE_MAX}) : {n_edges:,}",
    f"work groups after union-find : {n_groups:,}",
    f"deduped top-K written        : {len(pdf)}",
    f"top exact Jaccard (deduped)  : {pdf['exact_jaccard'].iloc[0]:.4f}" if len(pdf) else "",
    f"100th exact Jaccard          : {pdf['exact_jaccard'].iloc[-1]:.4f}" if len(pdf) == TOP_K else
        f"100th not reached (only {len(pdf)} distinct work-pairs in input)",
]
(spark.createDataFrame([(l,) for l in summary if l], ["value"])
      .coalesce(1)
      .write.mode("overwrite")
      .text(f"{OUT_DIR}/summary_txt"))
for line in summary:
    if line:
        print(line)

elapsed = time.time() - t0
print(f"\n[Step 2b complete in {elapsed/60:.1f} min]")
spark.stop()

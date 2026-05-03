
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALSModel

spark = (SparkSession.builder.appName("D3_Debug5").getOrCreate())
spark.sparkContext.setLogLevel("WARN")

BASE      = "hdfs:///user/at7095_nyu_edu/goodreads"
OUT_DIR   = f"{BASE}/als_explicit/full"

# Check val_small_preds recs sizes
preds = spark.read.parquet(f"{OUT_DIR}/val_small_preds")
print(f"Total users with preds: {preds.count()}")
preds.withColumn("recs_len", F.size("recs")).agg(
    F.min("recs_len").alias("min"),
    F.mean("recs_len").alias("avg"),
    F.max("recs_len").alias("max")
).show()

# Check val_gt
val_small = spark.read.parquet(f"{BASE}/val_small.parquet")
val_gt = (val_small.filter(F.col("is_read") == 1)
                   .groupBy("user_id")
                   .agg(F.collect_list("book_id").alias("ground_truth")))
print(f"Val GT users: {val_gt.count()}")

# Check overlap between preds user_ids and val_gt user_ids
overlap = preds.join(val_gt, on="user_id", how="inner")
print(f"Overlapping users: {overlap.count()}")

# Show a sample
overlap.show(3, truncate=False)

spark.stop()

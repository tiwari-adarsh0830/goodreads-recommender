
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (SparkSession.builder
         .appName("D3_Debug")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

BASE    = "hdfs:///user/at7095_nyu_edu/goodreads"
OUT_DIR = f"{BASE}/als_explicit/full"

val_preds = spark.read.parquet(f"{OUT_DIR}/val_preds")
val_gt    = spark.read.parquet(f"{OUT_DIR}/val_gt_tmp")

print(f"val_preds count: {val_preds.count()}")
print(f"val_gt count: {val_gt.count()}")

# Check a sample prediction
val_preds.show(3, truncate=False)
val_gt.show(3, truncate=False)

# Check overlap
overlap = val_preds.join(val_gt, on="user_id", how="inner")
print(f"Overlapping users: {overlap.count()}")

spark.stop()

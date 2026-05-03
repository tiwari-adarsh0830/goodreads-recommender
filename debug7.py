
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("debug7").getOrCreate()
spark.sparkContext.setLogLevel("WARN")
BASE = "hdfs:///user/at7095_nyu_edu/goodreads"

# Check what books small split users actually interacted with
val_small = spark.read.parquet(f"{BASE}/val_small.parquet")
train_small = spark.read.parquet(f"{BASE}/train_small.parquet")

print("=== val_small is_read=1 book_id range ===")
val_small.filter(F.col("is_read")==1).agg(F.min("book_id"), F.max("book_id"), F.countDistinct("book_id")).show()

print("=== train_small book_id range ===")
train_small.agg(F.min("book_id"), F.max("book_id"), F.countDistinct("book_id")).show()

# Check if small split user_ids exist in full train
train_full = spark.read.parquet(f"{BASE}/train.parquet")
small_users = train_small.select("user_id").distinct()
overlap = small_users.join(train_full.select("user_id").distinct(), on="user_id", how="inner")
print(f"Small users in full train: {overlap.count()} / {small_users.count()}")

spark.stop()

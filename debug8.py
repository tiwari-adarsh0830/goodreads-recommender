
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALSModel

spark = SparkSession.builder.appName("debug8").getOrCreate()
spark.sparkContext.setLogLevel("WARN")
BASE = "hdfs:///user/at7095_nyu_edu/goodreads"

val_small = spark.read.parquet(f"{BASE}/val_small.parquet")
train_small = spark.read.parquet(f"{BASE}/train_small.parquet")

# For user 24, what are their ground truth books?
gt_books = val_small.filter((F.col("user_id")==24) & (F.col("is_read")==1)).select("book_id", "rating")
print("=== User 24 ground truth books (is_read=1) ===")
gt_books.show(10)

# What did the model recommend for user 24?
model = ALSModel.load(f"{BASE}/als_explicit/full/model")
preds = spark.read.parquet(f"{BASE}/als_explicit/full/val_small_preds")
print("=== User 24 recommendations (first 10) ===")
preds.filter(F.col("user_id")==24).select(F.col("recs")[0:10]).show(truncate=False)

# Do any recommended books appear in ground truth?
user24_recs = preds.filter(F.col("user_id")==24).select(F.explode("recs").alias("book_id"))
user24_gt = val_small.filter((F.col("user_id")==24) & (F.col("is_read")==1)).select("book_id")
hit = user24_recs.join(user24_gt, on="book_id", how="inner")
print(f"=== User 24 hits: {hit.count()} ===")

spark.stop()

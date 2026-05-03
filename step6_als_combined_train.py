
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS

spark = (SparkSession.builder
         .appName("D5_ALS_Combined_Train")
         .config("spark.sql.autoBroadcastJoinThreshold", "-1")
         .config("spark.driver.maxResultSize", "4g")
         .config("spark.sql.shuffle.partitions", "400")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

BASE    = "hdfs:///user/at7095_nyu_edu/goodreads"
OUT_DIR = f"{BASE}/als_combined/full"

print("[train] Loading training data...")
train = spark.read.parquet(f"{BASE}/train.parquet")

# Combined signal:
# - Start with is_read as base (implicit consumption)
# - If rated, use normalized rating (1-5 -> 0.2-1.0) as preference strength
# - Boost by is_reviewed (engagement signal)
# combined = (rating/5.0 if rating>0 else 0.5) * is_read + 0.3 * is_reviewed
# Then use ALS implicit mode with this as confidence signal

train_combined = train.withColumn(
    "combined_score",
    F.when(F.col("rating") > 0,
        (F.col("rating") / 5.0) * F.col("is_read").cast("float") +
        0.3 * F.col("is_reviewed").cast("float")
    ).otherwise(
        0.5 * F.col("is_read").cast("float") +
        0.3 * F.col("is_reviewed").cast("float")
    )
).filter(F.col("combined_score") > 0).select("user_id", "book_id", "combined_score")

print(f"[train] Training rows: {train_combined.count():,}")

als = ALS(
    rank=50,
    maxIter=10,
    regParam=0.1,
    alpha=40.0,
    implicitPrefs=True,
    userCol="user_id",
    itemCol="book_id",
    ratingCol="combined_score",
    coldStartStrategy="drop",
    seed=1004,
    nonnegative=False,
)

print("[train] Training ALS combined model...")
model = als.fit(train_combined)
print(f"[train] Training complete. rank={model.rank}")

model.save(f"{OUT_DIR}/model")
print(f"[train] Model saved -> {OUT_DIR}/model")
spark.stop()

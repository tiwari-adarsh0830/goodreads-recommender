
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS

spark = (SparkSession.builder
         .appName("D4_ALS_Implicit_v2_Train")
         .config("spark.sql.autoBroadcastJoinThreshold", "-1")
         .config("spark.driver.maxResultSize", "4g")
         .config("spark.sql.shuffle.partitions", "400")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

BASE    = "hdfs:///user/at7095_nyu_edu/goodreads"
OUT_DIR = f"{BASE}/als_implicit_v2/full"

print("[train] Loading training data...")
train = spark.read.parquet(f"{BASE}/train.parquet")

# Implicit signal: is_read + 0.5 * is_reviewed
# is_read = consumption signal
# is_reviewed = engagement signal (additional confidence)
# Keep only interactions where user read or reviewed
train_implicit = train.withColumn(
    "implicit_signal",
    F.col("is_read").cast("float") + 0.5 * F.col("is_reviewed").cast("float")
).filter(F.col("implicit_signal") > 0).select("user_id", "book_id", "implicit_signal")

print(f"[train] Training rows: {train_implicit.count():,}")

als = ALS(
    rank=50,
    maxIter=10,
    regParam=0.1,
    alpha=40.0,
    implicitPrefs=True,
    userCol="user_id",
    itemCol="book_id",
    ratingCol="implicit_signal",
    coldStartStrategy="drop",
    seed=1004,
    nonnegative=False,
)

print("[train] Training ALS implicit v2 model...")
model = als.fit(train_implicit)
print(f"[train] Training complete. rank={model.rank}")

model.save(f"{OUT_DIR}/model")
print(f"[train] Model saved -> {OUT_DIR}/model")
spark.stop()


import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS

spark = (SparkSession.builder
         .appName("D4_ALS_Implicit_Train")
         .config("spark.sql.autoBroadcastJoinThreshold", "-1")
         .config("spark.driver.maxResultSize", "4g")
         .config("spark.sql.shuffle.partitions", "400")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

BASE      = f"hdfs:///user/{os.environ.get('NETID', 'your_netid')}_nyu_edu/goodreads"
OUT_DIR   = f"{BASE}/als_implicit/full"

print("[train] Loading training data...")
train = spark.read.parquet(f"{BASE}/train.parquet")

# Use ALL interactions, is_read as confidence signal
# ALS implicitPrefs uses: confidence = 1 + alpha * is_read
# is_read is 0 or 1, so confidence = 1 or 1+alpha
train_implicit = train.select("user_id", "book_id", F.col("is_read").cast("float").alias("is_read"))

print(f"[train] Training rows: {train_implicit.count():,}")

als = ALS(
    rank=50,
    maxIter=10,
    regParam=0.1,
    alpha=40.0,
    implicitPrefs=True,
    userCol="user_id",
    itemCol="book_id",
    ratingCol="is_read",
    coldStartStrategy="drop",
    seed=1004,
    nonnegative=False,
)

print("[train] Training ALS implicit model...")
model = als.fit(train_implicit)
print(f"[train] Training complete. rank={model.rank}")

model.save(f"{OUT_DIR}/model")
print(f"[train] Model saved -> {OUT_DIR}/model")
spark.stop()


from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (SparkSession.builder.appName("D3_Debug4").getOrCreate())
spark.sparkContext.setLogLevel("WARN")

BASE = "hdfs:///user/at7095_nyu_edu/goodreads"
OUT_DIR = f"{BASE}/als_explicit/full"

train_seen = spark.read.parquet(f"{OUT_DIR}/train_seen_tmp")
stats = train_seen.groupBy("user_id").count().agg(
    F.min("count").alias("min_books"),
    F.mean("count").alias("avg_books"),
    F.percentile_approx("count", 0.5).alias("median_books"),
    F.percentile_approx("count", 0.9).alias("p90_books"),
    F.max("count").alias("max_books"),
)
stats.show()
spark.stop()

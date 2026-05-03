
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (SparkSession.builder.appName("D3_Debug3").getOrCreate())
spark.sparkContext.setLogLevel("WARN")

BASE    = "hdfs:///user/at7095_nyu_edu/goodreads"
OUT_DIR = f"{BASE}/als_explicit/full"

train_seen = spark.read.parquet(f"{OUT_DIR}/train_seen_tmp")
print(f"train_seen count: {train_seen.count()}")
train_seen.show(5)
spark.stop()

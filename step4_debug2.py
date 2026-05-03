
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (SparkSession.builder.appName("D3_Debug2").getOrCreate())
spark.sparkContext.setLogLevel("WARN")

BASE    = "hdfs:///user/at7095_nyu_edu/goodreads"
OUT_DIR = f"{BASE}/als_explicit/full"

val_preds = spark.read.parquet(f"{OUT_DIR}/val_preds")

# Check recs array sizes
val_preds.withColumn("recs_len", F.size("recs")).groupBy("recs_len").count().orderBy("recs_len").show(20)

# Check how many files in val_preds (= how many batches completed)
spark.stop()

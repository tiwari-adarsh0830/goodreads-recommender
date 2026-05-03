
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALSModel

spark = SparkSession.builder.appName("debug6").getOrCreate()
spark.sparkContext.setLogLevel("WARN")
BASE = "hdfs:///user/at7095_nyu_edu/goodreads"

print("=== train_small book_id range ===")
spark.read.parquet(f"{BASE}/train_small.parquet").agg(F.min("book_id"), F.max("book_id")).show()
print("=== val_small book_id range ===")
spark.read.parquet(f"{BASE}/val_small.parquet").agg(F.min("book_id"), F.max("book_id")).show()
print("=== train.parquet book_id range ===")
spark.read.parquet(f"{BASE}/train.parquet").agg(F.min("book_id"), F.max("book_id")).show()
print("=== model itemFactors id range ===")
model = ALSModel.load(f"{BASE}/als_explicit/full/model")
model.itemFactors.agg(F.min("id"), F.max("id")).show()
spark.stop()

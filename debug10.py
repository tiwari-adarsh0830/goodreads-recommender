
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALSModel
import numpy as np

spark = SparkSession.builder.appName("debug10").getOrCreate()
spark.sparkContext.setLogLevel("WARN")
BASE = "hdfs:///user/at7095_nyu_edu/goodreads"

model = ALSModel.load(f"{BASE}/als_explicit/full/model")

# Get user 24 factor
u_vec = np.array(model.userFactors.filter(F.col("id")==24).collect()[0]["features"])

# Get scores for GT books
gt_ids = [13979, 15316, 15259, 15312, 15252, 15138, 12625, 943, 15102, 15107]
gt_rows = model.itemFactors.filter(F.col("id").isin(gt_ids)).collect()
print("Scores for GT books:")
for r in gt_rows:
    score = float(np.dot(u_vec, np.array(r["features"])))
    print(f"  book {r['id']}: {score:.4f}")

# Get scores for recommended books (first 10 from val_small_preds)
preds = spark.read.parquet(f"{BASE}/als_explicit/full/val_small_preds")
rec_ids = preds.filter(F.col("user_id")==24).select(F.explode("recs").alias("book_id")).limit(10).collect()
rec_ids = [r["book_id"] for r in rec_ids]
rec_rows = model.itemFactors.filter(F.col("id").isin(rec_ids)).collect()
print("Scores for recommended books:")
for r in rec_rows:
    score = float(np.dot(u_vec, np.array(r["features"])))
    print(f"  book {r['id']}: {score:.4f}")

spark.stop()

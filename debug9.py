
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALSModel

spark = SparkSession.builder.appName("debug9").getOrCreate()
spark.sparkContext.setLogLevel("WARN")
BASE = "hdfs:///user/at7095_nyu_edu/goodreads"

model = ALSModel.load(f"{BASE}/als_explicit/full/model")

# Check if user 24 ground truth books exist in model item factors
gt_book_ids = [13979, 15316, 15259, 15312, 15252, 15138, 12625, 943, 15102, 15107]
gt_df = spark.createDataFrame([(b,) for b in gt_book_ids], ["id"])
in_model = gt_df.join(model.itemFactors.select("id"), on="id", how="inner")
print(f"GT books in model: {in_model.count()} / {len(gt_book_ids)}")

# Check if user 24 exists in model user factors
user_df = spark.createDataFrame([(24,)], ["id"])
in_model_user = user_df.join(model.userFactors.select("id"), on="id", how="inner")
print(f"User 24 in model: {in_model_user.count()}")

spark.stop()

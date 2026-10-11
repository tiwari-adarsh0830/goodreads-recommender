"""
eval_helpers.py — canonical ranking-metric evaluator for the Goodreads capstone.

Author: Adarsh Tiwari and Sheetal Sood (Goodreads Capstone, DSGA1004 Big Data, Spring 2026)

Used by deliverables 2, 3, 4, 5 (popularity, explicit ALS, implicit ALS,
combined). The single function `compute_ranking_metrics` accepts a
predictions DataFrame [user_id, recs (array<int>)] and a held-out DataFrame
[user_id, ground_truth (array<int>)] and returns Precision@K, MAP@K, NDCG@K
via Spark's RankingMetrics.

Why one shared helper
---------------------
The brief states: "Your popularity baseline should be evaluated on the
validation and test data using the same general evaluation framework you
will use for later models." Reusing this single function across four
deliverables guarantees we are not silently changing the evaluation
methodology between models -- comparison numbers are apples-to-apples.

Correctness rules enforced here
-------------------------------
1. Top-K predictions per user (K=100 by default, per brief).
2. Ground truth = a per-user array of held-out item IDs supplied by caller.
3. Users with empty ground truth are dropped (cannot evaluate them).
4. Users with empty predictions are dropped (recommender produced nothing).
5. Inner join on user_id -- mismatched users are excluded automatically.
6. Spark's RankingMetrics expects RDD[(predicted: list, actual: list)].
"""
from pyspark.sql import functions as F
from pyspark.mllib.evaluation import RankingMetrics


def compute_ranking_metrics(predictions, ground_truth, k=100,
                             pred_col="recs", truth_col="ground_truth",
                             user_col="user_id"):
    """
    Evaluate top-K recommendations against per-user ground truth.

    Parameters
    ----------
    predictions  : Spark DataFrame with columns [user_col, pred_col]
                   where pred_col is array<int> of recommended item IDs,
                   most-relevant first, length <= k.
    ground_truth : Spark DataFrame with columns [user_col, truth_col]
                   where truth_col is array<int> of held-out item IDs.
    k            : evaluation cutoff (default 100, per the brief).

    Returns
    -------
    dict with keys:
        n_users_eval     -- number of users actually scored
        precision_at_k
        map_at_k
        ndcg_at_k
    """
    # Inner join: only users present in BOTH sides are scored.
    paired = (predictions.alias("p")
              .join(ground_truth.alias("g"), user_col, "inner")
              .select(F.col(f"p.{pred_col}").alias("preds"),
                      F.col(f"g.{truth_col}").alias("actuals")))

    # Drop empty-prediction or empty-ground-truth users.
    paired = paired.filter(
        (F.size("preds") > 0) & (F.size("actuals") > 0))

    # Truncate predictions to k (caller may have given more).
    paired = paired.withColumn("preds", F.slice("preds", 1, k))

    n_users_eval = paired.count()
    if n_users_eval == 0:
        raise ValueError("No users left to evaluate after filtering "
                         "empty predictions/ground truth.")

    # RankingMetrics expects RDD[(predicted_list, ground_truth_list)].
    rdd = (paired.rdd
           .map(lambda r: (list(r["preds"]), list(r["actuals"]))))

    rm = RankingMetrics(rdd)
    return {
        "n_users_eval"  : n_users_eval,
        "precision_at_k": rm.precisionAt(k),
        "map_at_k"      : rm.meanAveragePrecisionAt(k),
        "ndcg_at_k"     : rm.ndcgAt(k),
    }


def format_metrics(metrics, label, k=100):
    """Pretty-print metrics dict for logs and report artifacts."""
    return (f"--- {label} (k={k}) ---\n"
            f"users evaluated   : {metrics['n_users_eval']:>10,}\n"
            f"Precision@{k:<3}    : {metrics['precision_at_k']:.6f}\n"
            f"MAP@{k:<3}          : {metrics['map_at_k']:.6f}\n"
            f"NDCG@{k:<3}         : {metrics['ndcg_at_k']:.6f}\n")

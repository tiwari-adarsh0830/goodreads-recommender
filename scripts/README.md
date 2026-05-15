# scripts/

Follow-up diagnostic scripts added after the initial pipeline.
The original step0–step6 pipeline lives at the repository root.

- `step2_dedup_editions.py` — collapses likely duplicate Goodreads editions
  in the top MinHash pairs using union-find on the high-Jaccard candidate
  set, then re-extracts the top-K most similar pairs across distinct
  works. Writes to `hdfs:///user/<netid>/goodreads/minhash/deduped/`.

- `step4_als_explicit_eval_rated_batched.py` — loads the already-saved
  D3 explicit ALS model and evaluates it against a rated-only ground
  truth (`is_read = 1 AND rating > 0`) in user batches to stay inside
  the YARN job-lifetime limit. Writes to
  `hdfs:///user/<netid>/goodreads/als_explicit/full/rated_gt_metrics_summary.txt`.

- `step7_variance_bands.py` — bootstrap 95% confidence intervals for
  P@100, MAP@100, and NDCG@100 on the small-split evaluation. Reads
  saved per-user predictions, joins to the small test/val split, and
  resamples users with replacement (B=500). CLI:
  `--model {d2,d4,d4v2,d5} --split {val,test} --b 500`.

- `step5_als_implicit_tune.py` — D4 implicit-ALS hyperparameter sweep
  (rank × regParam × alpha grid). Implemented but **not run**; deferred
  due to NYU Dataproc compute limits. See the Future Work section of
  the report.

All scripts import `eval_helpers.py` from the repository root and
expect to be submitted with `--py-files /path/to/eval_helpers.py`.

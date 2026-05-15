# results/

Output files from the follow-up diagnostic runs in `scripts/`.

- `top100_pairs_deduped.csv` — deduplicated top MinHash pairs after
  edition-collapsing. The original top-100 from `step2_top100_pairs.csv`
  reduces to 37 distinct work-pairs once likely duplicate editions are
  unioned. Top exact Jaccard after dedup: 0.9328.

- `rated_gt_metrics_summary.txt` — rated-only D3 evaluation
  (`is_read = 1 AND rating > 0` ground truth). Validation: 7,497 users,
  MAP@100 ≈ 1e-5. Test: 7,536 users, MAP@100 ≈ 1e-6. Even when scored
  against the target explicit ALS was actually optimized for, the
  metrics remain effectively zero — confirming that explicit ratings
  alone are too sparse/weak for top-100 recommendation under this
  pipeline.

- `variance_d4_test.txt` — bootstrap 95% CIs for D4 implicit ALS on the
  test split. 7,699 users, B=500. MAP@100 point=0.0737, 95% CI
  [0.0715, 0.0764].

- `variance_d5_test.txt` — bootstrap 95% CIs for D5 combined ALS on the
  test split. 7,683 users, B=500. MAP@100 point=0.0765, 95% CI
  [0.0740, 0.0788].

The D4/D5 MAP@100 confidence intervals overlap, so the D5 improvement
over D4 should be read as directionally consistent rather than
statistically decisive on the 1% sample.

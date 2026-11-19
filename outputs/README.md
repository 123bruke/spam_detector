# Generated training outputs

The files currently included here were produced by running `python train.py --demo` because the build environment had no external network access. The included metrics and plots are **pipeline smoke-test results only**. Both demo models happen to score 100% on a very small, synthetic, deliberately separable test split; this is not a benchmark or a claim of real-world performance.

Run `python train.py` with network access to fetch the full UCI SMS Spam Collection and regenerate `metrics.json`, `test_predictions.csv`, `training_history.csv`, and the six charts.

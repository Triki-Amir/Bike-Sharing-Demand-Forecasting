# Use of AI coding assistants

| Part | Assistant | What it did | What I checked or changed |
|---|---|---|---|
| Project scaffold, `src/bikecast` modules | Claude Code | First version of the config, metrics, segmentation, Markov, ZIP and backtest modules from the project plan | Ran every function on real data; reviewed cutoff logic so training never sees test days |
| Local pandas pipeline (`pipeline.py`) | Claude Code | Translated the Databricks SQL (silver, gold, segmentation) to pandas so the project runs and is tested without Databricks | Matched the SQL semantics exactly: dedup ordering (Spark sorts NULLs first), `unix_timestamp` second truncation, `dayofweek` with 1 = Sunday |
| Spark SQL tests | Claude Code | Test that runs the notebook SQL on a local Spark session and compares it with pandas row for row | Session time zone set to UTC to match Databricks (DST would otherwise shift trip durations) |
| Synthetic data generator | Claude Code | Generates trip files in the real format with known segments and dirty rows (duplicates, dockless, too short, a new station) | Used as ground truth in tests: silver must remove exactly the injected rows; segments must match |
| CI/CD | Claude Code | GitHub Actions (lint, tests, Spark SQL, synthetic end-to-end) and the Databricks Asset Bundle + deploy workflow | Deploy is skipped cleanly when Databricks secrets are absent |
| Charts | Claude Code | Matplotlib charts with a colour-blind-checked palette; no dual axes (P(trip) is its own panel) | Looked at every figure before committing |
| Notebooks | Databricks Assistant (optional) | Small SQL tweaks while running in the workspace | — |

Lessons: the assistant was fast at boilerplate and at keeping SQL and pandas versions in sync, but every
modelling choice (training-only segmentation, tuning on the first two cutoffs only, MASE scaling) was checked
by hand against the backtest design.

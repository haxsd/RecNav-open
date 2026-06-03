# scripts

Release entrypoints:

- `scripts/core/check_dataset.py`: validate local HM3D/ObjectNav paths.
- `scripts/core/smoke_test.py`: run a small Habitat reset/step check.
- `scripts/pixnav/run_pixnav_host_eval.py`: main PixNav + RecNav evaluator.
- `scripts/pixnav/analyze_results.py`: summarize available result CSVs.
- `scripts/pixnav/plot_budget_curve.py`: plot SR/SPL budget curves.
- `scripts/tools/stage0_env_check.py`: verify imports, paths, and checkpoints.
- `scripts/tools/llm_smoke.py`: check OpenAI-compatible LLM connectivity.
- `scripts/tools/mcnemar_test.py`: paired McNemar exact test.

Use `./run.sh` from the repository root for the common commands.

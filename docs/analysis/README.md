# Analysis (2026-09-16)

Read-only reverse-engineering of the `develop` branch. No application code was changed.

- `REVERSE_ENGINEERING_REPORT.md` — the consolidated report (executive summary, architecture, Git archaeology, data audit, model inventory, execution flow, what works / what is broken / broken by design, reliability of results, root causes, literature, proposed redesign, evaluation framework, roadmap, priority fixes, open questions, five answers).
- `appendix/` — the six underlying investigation reports and the GitHub issue/PR evidence.
- `repro/` — throwaway scripts used to reproduce findings (look-ahead perturbation tests, multi-timeframe leak, scaler leakage, synthetic fallback, live-env buffer, backtester timestamps, train/inference mismatch, naive-baseline loss). They import repository modules and need `pandas numpy scikit-learn torch gymnasium tqdm`; run from the repository root, e.g. `python docs/analysis/repro/t7_mismatch.py`.

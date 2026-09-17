# GitHub issue/PR evidence (fetched via API, 2026-09-16)

Timeline: ENTIRE project built 2025-12-07 -> 2025-12-18 (11 days), 107 PRs/issues, nearly all branches `claude/...` (AI-generated), CodeRabbit reviews.
Issues 103,104,105,106,107 closed TODAY (2026-09-16) as not_planned => never fixed. Code on develop still has these bugs.

Key evidence:
- PR #1 (Dec 7): initial 7,660-line drop: TFT + PPO + online learning + backtester + risk mgmt. 19 commits, one session.
- PR #5 (Dec 8): backtester traded 49,792 of 50,000 bars; 2% risk compounding => 10^56 % return. "Fixed" by adding --realistic cooldown/max-daily-trades constraints, not by fixing the signal.
- Issue #12 (Dec 9): `live --paper` was a stub: infinite sleep loop with placeholder comments.
- PR #43 (Dec 10): walk-forward was a stub: train_func returned None, strategy always 'hold'. Rewritten to train a fresh predictor per fold, threshold-based signal on predicted return.
- PR #54 (Dec 11): PPO rewards only negative because max_drawdown (monotonic) penalised every step. Changed to delta-drawdown. "OLD 9% positive rewards mean -0.64 -> NEW 46% positive mean -0.12".
- PR #64 (Dec 12): "12 potential bugs in PPO": reward scaling, ratio overflow, KL NaN, GAE in online learning, dtype mismatch, unbounded logits, advantage normalization. Docs PPO_AGENT_ANALYSIS.md, PPO_FIXES_APPLIED.md, PPO_TUNING_GUIDE.md (later deleted).
- PR #67 (Dec 13): CRITICAL env bug: reward computed at the SAME price used for entry (no time advance) => agent learned to avoid trading because only costs were visible. Fixed by advancing step before mark-to-market. Also: SL/TP double-spread, OHLC-based SL/TP detection, margin enforcement.
- PR #68 (Dec 13): autotrade never called data_pipeline.connect() => SILENTLY USED SYNTHETIC DATA. OnlineLearningManager never initialised.
- PR #69 (Dec 13): reward v1 parametrisation: return_scale 50, dd penalty 20, recovery bonus 20, holding cost 0, clip 5. utils/reward_analyzer.py added (later deprecated PR #97).
- PR #74 (Dec 15): reward v2: r = log(E_t/E_{t-1}) - c_tc*|dw| - lambda_dd*max(0, DD-DD_max) - lambda_vol*sigma^2; Welford running reward normalisation in PPO.
- PR #78 (Dec 17): backtester made to use the same Transformer+PPO combination as AutoTrader._combine_signals with confidence thresholds.
- Issue #81 (Dec 17): backtest loads checkpoint with wrong d_model (uses current config not saved config).
- PR #84 (Dec 17): "ZERO TRADES BUG": default prediction_confidence=0.5 < min_confidence=0.6 filtered every signal to 'hold'. Fix: hard-code prediction_confidence=0.7 (a constant! not a real confidence). Also float32 overflow in account obs -> log1p normalisation; PPO eval_split added.
- Issue #101 (Dec 18): sliding_window_view TRANSPOSE BUG: training sequences were (samples, features, seq_len) so model input_dim = seq_len (192) not n_features (92). EVERY model trained before PR #102 saw axes swapped. Also feature-count mismatch: training uses OHLCV+computed (92), strategy.py excludes OHLCV (87). Multi-timeframe also affected (159/164 features).
- Issue #104 (Dec 18, NOT FIXED): adapt command still uses 87 features vs model's 92 => crashes.
- Issue #105 (Dec 18, NOT FIXED): auto_trader.py:628 calls position_sync.get_all_positions() which doesn't exist => ALWAYS falls back to "legacy" signal path; CombinedPredictorAgentStrategy never actually runs in autotrade.
- Issue #103 (NOT FIXED): --no-mlflow ignored in backtest.
- Issue #106 (NOT FIXED): PPO hidden_sizes needs >=3 entries undocumented.
- Issue #107 (Dec 18, NOT FIXED): walkforward ALWAYS 0 trades: _wf_strategy threshold +-0.001 (0.1%) on predicted return; predictions rarely exceed it. "Walk-forward optimization is completely non-functional".
- Historic result files existed: results/backtest_20251208_142246.json, _145254, 20251210_174332, 20251210_180304 (deleted in b6142bb). Also deleted: docs/ARCHITECTURE_AUDIT.md, ARCHITECTURE_MISMATCH_REPORT.md, CODE_DUPLICATION_REPORT.md, docs/PPO_*.md, tests/test_ppo_bugs_evidence.py, tests/test_ppo_agent.py.

# Independent verification by lead (2026-09-16)
VERIFIED (read code myself):
1. core/data_pipeline.py:1006-1011 RobustScaler.fit_transform on FULL feature array (before split). grep for scaler/transform in strategy.py, auto_trader.py, system.py, backtester.py, live_trading_env.py => ZERO hits. Inference feeds raw values.
2. core/trading_env.py:247-258: obs = concat(prices[t-w:t].flatten(), features[t-w:t].flatten()), then global z-score; excludes bar t. core/strategy.py:~483-494: market_data[cols].tail(w).values.flatten() (row-major interleave), no z-score. Layout+scale mismatch confirmed.
3. core/auto_trader.py: only env.get_market_data() at L621; env.step never called; live_trading_env.py:568 timeframe='1h' hard-coded; buffer filled only in _fetch_initial_data at reset.
4. trading_types.py: tc_penalty_scale=1.0, max_position_size=0.1, spread 2e-4, slippage 1e-4, commission 1e-4. Reward at trading_env_base.py:266-343 exactly as reported.
5. evaluation/backtester.py:258 timestamp = datetime.now() when index not datetime; cli/system.py:757-763 builds DataFrame with default RangeIndex => all bars get wall-clock timestamps.
6. config/templates/training.json ppo.total_timesteps=10000 (n_steps 2048 => ~4 PPO updates). transformer.max_seq_length=120 in template but data.json lookback_window=192.

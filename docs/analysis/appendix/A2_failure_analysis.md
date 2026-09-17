# Leap — Read-only failure analysis

Repo: `/home/user/Leap` @ `claude/reverse-engineer-redesign-jy7zdo` (HEAD aa79be6). No repo file was modified (`git status --porcelain` empty). Tests and verification scripts were run against a scratch copy in an isolated venv (torch 2.14, pandas 3.0.5, numpy pinned <2 per requirements).

Legend: **FACT** = read from code and/or reproduced by script; **INFERENCE** = consequence I derive but did not execute end-to-end (e.g. needs MT5 or a trained model). Verification scripts: `scratchpad/verify_more.py`, `scratchpad/verify_8_10.py`.

## Test suite

`python -m pytest -q` → **321 passed, 3 failed** (24 s).
Failures are all in `tests/test_integration.py` (`test_backtester`, `test_monte_carlo_integration`, `test_full_integration`): `pd.date_range(..., freq='H')` at `tests/test_integration.py:297` and `:467` — the `'H'` alias was removed in pandas 3 (requirements say `pandas>=2.0.0`, unpinned). Several integration tests `return` values instead of asserting (PytestReturnNotNoneWarning).

What the tests cover vs. mock away (FACT):
- `tests/test_cli.py` (47 tests, 71 Mock refs, 27 patches) replaces `DataPipeline`, `TransformerPredictor`, `PPOAgent`, `ModelTrainer` with stubs (`MockPredictor.predict` returns a constant dict). Nothing about ML correctness or model/data plumbing is exercised.
- `tests/test_strategy.py` uses `Mock` predictors/agents; asserts signal-combination logic only. No test checks that inference features are scaled like training features, or that `positions` may be `BrokerPosition`.
- `tests/test_integration.py` builds tiny real models and asserts shapes / “did not crash”. The backtester test uses a proper `DatetimeIndex`, so the CLI’s RangeIndex path (F6) is never tested.
- No test covers `OrderManager` + `RiskManager` sizing (F5), the AutoTrader loop refreshing data (F3), online learning being fed from AutoTrader (F9), or walk-forward training (F7).

---

## Executive summary

The system is internally inconsistent along every seam that matters for a trading system: the models are trained on scaled/normalised inputs but served raw inputs; the backtester reports risk metrics computed from wall-clock timestamps; walk-forward silently trains a mis-shaped model and then trades nothing; the live loop never refreshes market data after start-up; position sizing confuses units and lots; the paper broker in the CLI path trades against frozen hard-coded prices; and the online-learning loop can never store an RL transition. Reported backtest performance cannot be trusted, and live/paper trading cannot behave as the backtest suggests. Most of these are fixable bugs, but F12 (three divergent execution models) and F10/F11 (in-sample evaluation, same-bar fills) are design flaws.

---

## CRITICAL

### F1 — Predictor trained on RobustScaler-scaled features, served raw features (train/inference distribution mismatch), scaler never persisted, scaler fit on the full dataset
- Category: wiring / leakage. **FACT**. Bug (fixable) + design gap (no artefact for the scaler).
- `core/data_pipeline.py:1003-1012`:
  ```python
  scaler_key = f"{data.symbol}_{data.timeframe}"
  if scaler_key not in self.scalers:
      self.scalers[scaler_key] = RobustScaler()
      features = self.scalers[scaler_key].fit_transform(features)   # fit on ALL bars, before split
  ```
  `create_train_val_test_split` (`:1052-1066`) splits *after* this, so the scaler sees val/test.
- Inference never scales: `cli/system.py:758-773` builds the backtest DataFrame directly from `market_data.open/.../features` (raw); `core/strategy.py:431` `features = market_data[feature_cols].tail(...).values` → `predictor.predict(X)` (`:452`). Live: `LiveTradingEnvironment.get_market_data()` (`core/live_trading_env.py:244-265`) is raw too.
- The scaler is not saved by `save_models`/`_save_predictor_only` (`cli/system.py:661-698,1069-1111`) and `load_models` never rebuilds one. `DataPipeline.inverse_transform_predictions` (`core/data_pipeline.py:1110-1120`) returns its input in both branches (no-op, dead).
- Impact: every prediction in backtest and live is out-of-distribution relative to training (prices ~1.1 and volumes ~1e3 instead of robust-scaled ~0). Backtest numbers are meaningless; commit 8345156 “aligned” the strategy object but not the data contract. `evaluate` (`cli/commands/evaluate.py:429-433`) refits a *new* scaler on eval data, so its MAE is also not comparable.

### F2 — PPO agent observation: training normalises, strategy inference does not; and the training normalisation itself erases the price/feature signal
- Category: wiring. **FACT** (formula) + **FACT** (numerical demonstration). Bug + design flaw.
- Training env (`core/trading_env.py:247-258`):
  ```python
  market_obs = np.concatenate([price_window, feature_window])
  market_obs = (market_obs - np.mean(market_obs)) / (np.std(market_obs) + 1e-8)
  ```
  One z-score across a flattened vector that mixes prices (~1), volume (~1e3–1e4), RSI (0–100), returns (1e-3). Measured on the pipeline’s own synthetic data (60-bar window, 87 features): obs mean 1575, std 10783; **std of the z-scored close within the window = 3e-6** (raw 0.03). 293/300 OHLCV values and 5219/5220 feature values lie within |z|<0.1 — the agent sees a near-constant market vector plus 8 account features. Same formula in `core/live_trading_env.py:397-403`.
- Inference (`core/strategy.py:487-503`): `market_obs = features.flatten()` — **no normalisation at all** → agent receives raw magnitudes it never saw. `LiveTradingEnvironment._get_observation` also silently pads/truncates to the expected dim (`:657-662`), hiding any mismatch.
- AutoTrader does not pass `account_state` to the strategy (`core/auto_trader.py:635-640`), so `_build_account_observation` uses the hard-coded defaults `balance=equity=10000, pnl=0` (`core/strategy.py:531-539`) — account features are constants in live.

### F3 — Live/paper trading generates every signal from a data snapshot taken once at start-up
- Category: incomplete / wiring. **FACT** (grep-verified). Bug.
- `LiveTradingEnvironment._fetch_initial_data()` is called only from `reset()` (`core/live_trading_env.py:311`); `_update_data_buffer()` only from `step()` (`:362`). `AutoTrader` calls `env.reset()` once (`core/auto_trader.py:466`) and **never calls `env.step()`** (grep: no `.step(` on a live env anywhere in runtime code). `_generate_signal` reads `env.get_market_data()` (`:621`), i.e. the stale buffer.
- Also: `_fetch_initial_data` hard-codes `timeframe='1h'` and `n_bars=window_size+50` (`:567-571`), ignoring `AutoTraderConfig.timeframe`; with 110 bars the 200-period features (`sma_200, ema_200, close_sma_200_ratio`) remain NaN after ffill/bfill (measured 2.3% NaN) and are zeroed by `nan_to_num` — another train/serve shift. Even `step()`’s buffer update would be wrong: it appends a pseudo-bar `[bid, ask, bid, bid, 0.0]` per call (`:596-602`) and never appends features, after which `get_market_data()` drops features because buffer lengths differ (`:252`).
- Impact: the “live” system trades the same frozen window forever; the only thing that changes between cycles is nothing.

### F4 — In live trading the agent is silently disabled whenever a position is open
- Category: silent fallback / wiring. **FACT** (reproduced). Bug.
- `AutoTrader._generate_signal` passes `PositionSynchronizer.get_all_positions()` items (`BrokerPosition`) as `positions` (`core/auto_trader.py:628-629`). `CombinedPredictorAgentStrategy._build_account_observation` does `p.direction == 'long'` (`core/strategy.py:549-550`); `BrokerPosition` has `is_long/is_short` but no `direction` (`core/broker_interface.py:148-184`) → `AttributeError` → caught by `except Exception` at `strategy.py:383-384` → `agent_action = Action.HOLD`. Reproduced: agent `select_action` never called, signal HOLD although the stub agent wanted CLOSE.
- Impact: once any position exists for a symbol, the agent can never emit CLOSE; exits only via SL/TP. Backtest (which passes `Trade` objects) does not have this behaviour → backtest ≠ live.

### F5 — OrderManager treats RiskManager’s unit-denominated size as lots (≈100× over-sizing) and books asymmetric notionals
- Category: wiring / brittle. **FACT** (reproduced). Bug.
- `RiskManager.calculate_position_size` returns **units** (`risk_amount / risk_per_unit`, capped by `balance*0.02/price` → ~181 units on $10k) (`core/risk_manager.py:189-266`). `OrderManager._calculate_position_params` assigns it straight to `volume` and then clamps to `[volume_min, volume_max]` lots (`core/order_manager.py:439-457`). Reproduced with `DynamicRiskManager(10000)` and the paper broker: RiskManager → 92.1 units; OrderManager → **92.15 lots = 9.2 M units, $10.0 M notional on a $10 k account**; paper broker rejects for margin (`Required 99991.96, Available 10000`). Without a RiskManager the fallback gives 0.2 lots (also 10× the 1%-risk answer 0.02 lots because `_get_pip_value` ignores `volume` semantics).
- Exposure bookkeeping is asymmetric: open `notional = volume * price * contract_size` (`:254`), close `notional = position.volume * position.price_open` (`:330`) → `RiskManager.state.total_exposure` ratchets upward, driving `max_by_exposure` to 0 over time (`risk_manager.py:257-261`).
- Impact: with the default wiring (`cli/commands/autotrade.py:309` passes `system.risk_manager`) every entry is rejected by margin (system never trades), or, on an account with enough margin, opens a catastrophically oversized position.

### F6 — Backtester timestamps are wall-clock `datetime.now()`; all annualised metrics and time-based constraints are garbage
- Category: metric / wiring. **FACT** (reproduced). Bug.
- `cli/system.py:758-773` builds the backtest DataFrame with a default RangeIndex and no timestamp column (although `market_data.timestamp` exists). `Backtester.run` (`evaluation/backtester.py:258`):
  ```python
  timestamp = current_bar.name if isinstance(current_bar.name, datetime) else datetime.now()
  ```
  Reproduced on 600 synthetic bars: `infer_periods_per_year` → **3.3e11**; `sharpe = -1.58e4`, `annualized_return = -1.0`, `volatility = 1.5e3`, `avg_trade_duration = 4.6e-7 h`. With `--realistic`, `max_daily_trades=5` is keyed on `timestamp.date()` (`:261-264`) so **only 5 trades are allowed in the entire backtest** (reproduced: 12 signals → 5 trades). Commit 7b9b6ed’s frequency inference is defeated by the CLI path; `tests/test_metrics_frequency.py` only tests `MetricsCalculator` with proper timestamps.
- Same in `walk_forward_test` (`cli/system.py:883-896`).

### F7 — Walk-forward trains a model with swapped axes and then silently trades nothing
- Category: wiring / silent fallback. **FACT** (shape reproduced; downstream behaviour is deterministic from the code). Bug.
- `cli/system.py:930-931`:
  ```python
  X_train = sliding_window_view(train_features, window_shape=lookback_window, axis=0)
  X_train = X_train[:n_samples].copy()
  ...
  input_dim = X_train.shape[2]
  ```
  `sliding_window_view` returns `(n, n_features, lookback)` (`prepare_sequences` explicitly transposes at `core/data_pipeline.py:1039`; this path does not). Reproduced: `(881, 37, 120)` → `input_dim = 120`, i.e. the lookback, not 37. The predictor is trained with sequence length = n_features and 120 "features".
- In `wf_strategy` (`:1025-1029`) `len(available_cols) != expected_input_dim` (37 ≠ 120) → `return {'action': 'hold'}`; if dims happened to match, `predictor.predict(X)` with `(1,120,37)` would raise and be swallowed (`:1047-1048`). Any `train_func` exception also degrades to a hold strategy (`:986-988, 1000-1004`).
- Impact: `walkforward` prints per-fold results that are all zero-trade, and reports them as if valid. Also, WF trains only the predictor and uses a threshold rule, not the Transformer+PPO `CombinedPredictorAgentStrategy` — it validates a different strategy than the one backtested/traded (design flaw). `WalkForwardOptimizer.generate_splits` hard-codes `bars_per_day = 24` (`evaluation/backtester.py:707`).

### F8 — Paper trading in the CLI path runs against hard-coded static prices
- Category: fake. **FACT** (reproduced). Bug.
- `cli/commands/autotrade.py:235-241` sets `use_real_prices=True` but calls `create_broker('paper', config=paper_config)` without `mt5_broker`; `PaperBrokerGateway.__init__` (`core/paper_broker.py:183-188`) then selects `DefaultPriceProvider` (reproduced: `type(b._price_provider).__name__ == 'DefaultPriceProvider'`), whose prices are constants (`EURUSD (1.0850, 1.0852)`, `:60-71`; unknown symbols → `(1.0000, 1.0003)` or `(130.00, 130.03)`, `:106-118`).
- Combined with F3 and with the data pipeline’s synthetic fallback (F10a), “paper mode” on a machine without MT5 is a random-walk feature stream (seed 42) driving trades at a frozen price; every trade’s P&L is exactly −(spread+slippage+commission).

### F9 — Online learning in the live path can never learn (no RL transitions, nonsense targets, shape mismatch)
- Category: incomplete / fake. **FACT** (reproduced with stubs). Bug.
- `AutoTrader._on_position_closed` calls `online_manager.step(..., done=True)` without `log_prob`/`value` (`core/auto_trader.py:982-989`); `OnlineLearningManager.step` only stores a transition when both are present (`training/online_learning.py:172-181`). Reproduced: 120 closed trades → `agent.experience_buffer` length **0** → `_adapt_agent` never runs (`:293`).
- `actual_return = profit / entry_price` (`auto_trader.py:966-967`) divides account-currency P&L by a price → magnitude ~45 for a $50 profit; the "prediction error" |0.001−45| = 44.999 always exceeds `error_threshold=0.05`, so adaptation is *forced* on garbage.
- `market_data['features']` is the flattened PPO observation (`:975-976`); `_adapt_predictor` reshapes to `(n, 1, obs_dim)` (`online_learning.py:326-327`) and calls `predictor.online_update` → transformer expects `(batch, seq, 5+n_features)` → shape error → logged and swallowed (`auto_trader.py:933-934`). Reproduced call shape `(50, 1, 2228)`.
- `cli/commands/adapt.py` has the same class of bug: `_prepare_sequences` uses `features` only (no OHLCV) (`:155-156, 427-443`) so `input_dim` never matches a `train`-produced model (5+n_features); target is a 1-bar return vs. the 12-bar horizon used in training; online mode predicts on `features.reshape(1,1,-1)` (`training/online_learning.py:506-508`) — seq_len 1.

---

## HIGH

### F10 — Silent synthetic-data fallback in every command + in-sample evaluation everywhere
- Category: silent fallback / overfitting. **FACT**. Design flaw.
- `DataPipeline.connect()` returns False when MT5 is missing (`core/data_pipeline.py:589-593`); `LeapTradingSystem.load_data` ignores the return (`cli/system.py:139`); `fetch_historical_data` then generates a seeded random walk (`:654-668, 729-772`) at INFO level. `train`, `backtest`, `evaluate`, `walkforward` all "succeed" on noise. The `--save-data` label `data_source` is computed from `getattr(system.data_pipeline, 'broker_gateway', None)` — the attribute does not exist (verified `False`), so it is always "synthetic".
- `backtest` loads the same symbol/timeframe/`n_bars` (default 50000, `cli/parser.py:392`) as `train` and evaluates on it (`cli/commands/backtest.py:46-51`). No holdout. PPO early stopping uses the last 20% of the same series as `eval_env` (`cli/system.py:282-296`) and the predictor’s val split overlaps it. `evaluate` runs the agent on an env built from the whole dataset (`cli/commands/evaluate.py:426,437`).

### F11 — Execution look-ahead and warm-up leakage
- Category: leakage. **FACT**. Design flaw.
- Backtester decides with `data.iloc[:i+1]` (includes bar *i*’s close) and fills at `current_bar['close']` of the same bar (`evaluation/backtester.py:256-304`). Live fills at the next available tick. TradingEnvironment fills at close[t] and marks at close[t+1] (`core/trading_env.py:170-211`) — a third convention.
- `_process_raw_data` does `df.ffill().bfill()` (`core/data_pipeline.py:829`) after feature computation: back-filling the warm-up rows copies **future** feature values into the past (measured 8.2% of the first 200 feature rows are NaN before the fill). Same in `_add_multi_timeframe_features` (`:868-870, 934-935`).
- SL/TP intrabar: SL checked before TP with no path assumption (`backtester.py:541-555`) — conservative, fine; but SL/TP fills at exact level with no gap modelling.

### F12 — Three divergent execution/risk models (agent trained under mechanics it never trades under)
- Category: wiring. **FACT**. Design flaw.
- `TradingEnvironment` (PPO training): spread/slippage as *fractions of price* (`EnvConfig.spread=0.0002`, `core/trading_types.py:139-141`), SL/TP as **2%/4% of price** (≈200/400 pips on EURUSD, `:144-145`), size = 10% of balance/price (`core/trading_env.py:299-301`), no `max_positions` cap, one position per direction (`trading_env_base.py:255-261`).
- `Backtester`: SL/TP 50/100 *pips* applied multiplicatively (`backtester.py:401-406`), risk-based sizing (`:178-180`), `max_positions=5`, cooldown/daily caps only in realistic mode.
- Live (`OrderManager`): absolute pips via `point*10` (`order_manager.py:423-433`), RiskManager sizing (F5), broker SL/TP.
- `cli/system.py:242-244` feeds the env `commission=7/100000` (a $7/lot round-trip re-interpreted as a per-side rate) and `spread_pips*0.0001` (breaks for JPY).

### F13 — Spread/slippage/SL are multiplicative in price with a hard-coded `pip_value=0.0001`; no swap anywhere
- Category: brittle / metric. **FACT** (reproduced). Bug.
- `evaluation/backtester.py:94, 379-406`: `entry = price*(1 + spread_pips*pip_value/2 + slippage)`; `SL = entry*(1 − stop_pips*pip_value)`. Reproduced: EURUSD@1.1 requested 50-pip SL → **55 pips**; USDJPY@150 → **73 JPY-pips** SL and 1.9-pip entry cost (should be ~2 pips). Risk-based sizing (`utils/position_sizing.py:140-142`) uses the same multiplicative formula so $ risk stays right, but SL distances and TP/SL geometry are wrong and symbol-dependent.
- Swap/rollover is not modelled in Backtester, TradingEnvironment, or PaperBroker (`swap=0.0` constant, `core/paper_broker.py:397`); commission in PaperBroker only.

### F14 — "Confidence" gate compares a return-quantile spread to a probability threshold
- Category: brittle / wiring. **FACT** (formula, reproduced range) + **INFERENCE** (trained-model magnitude). Design flaw.
- `TransformerPredictor.predict` → `uncertainty = q90 − q10` of the *predicted return* (`models/transformer.py:669-673`); strategy uses `confidence = 1 − uncertainty` vs `min_confidence=0.6` (`core/strategy.py:366-368, 594`). For a trained model the 10–90 spread of 12-bar FX returns is ~1e-2 → confidence ≈ 0.99 → gate never fires; for the untrained stub it produced 0.45–0.80 (random gating). `OrderManager` has a second, independent `min_confidence=0.5` (`order_manager.py:97,365`).

### F15 — Bar timing in AutoTrader is wall-clock since start, not bar boundaries; interval ignores timeframe
- Category: brittle. **FACT**. Bug.
- `_is_new_bar` fires immediately on first call and then every `bar_interval` seconds from that arbitrary moment (`core/auto_trader.py:784-807`). `bar_interval` is fixed at 3600 (`config/settings.py:229`) and `autotrade.py:283-295` never sets it from the timeframe. `_check_daily_limits` reuses `_last_adaptation_date` for the daily reset (`:835-843`).

### F16 — RiskManager circuit breakers are never fed
- Category: dead / wiring. **FACT** (grep). Bug.
- `update_balance`, `record_trade`, `update_market_conditions` have **zero** runtime callers. Hence `current_drawdown`, `daily_pnl`, `consecutive_losses`, `regime` never change: drawdown/daily/weekly/consecutive-loss halts (`core/risk_manager.py:375-406`) are inert, Kelly sizing stays in the `< 20 trades` branch forever (`:232-234`), and `DynamicRiskManager` is always `'normal'` (`:478`). Only `on_position_opened/closed` are called (with the F5 asymmetry).

### F17 — Metrics: stateful `calculate_all`, inconsistent Sortino, Monte-Carlo compounding of notional returns
- Category: metric. **FACT** (reproduced). Bugs.
- `MetricsCalculator.calculate_all` sets `self.periods_per_year = ppy` (`evaluation/metrics.py:92-93`); later calls with different timestamps reuse it (`_resolve_periods_per_year` short-circuits at `:73-74`). Reproduced: hourly then daily on one instance → annualised 317.7 both times vs 0.27 on a fresh instance. `PerformanceAnalyzer` holds one instance.
- Sortino: `_sortino_from_returns` (`:181-189`, used by `calculate_all`) returns `inf` when there are no negative returns and mean>0; `sortino_ratio` (`:296-308`, used by Backtester) returns 0 in the same case, and both use `std(negatives)` rather than the target semi-deviation over all periods. `Sharpe`/`volatility`/`calmar` formulas themselves are algebraically fine; they are wrong in practice because of F6.
- `Trade.pnl_pct = pnl / (entry_price*size)` = return on **notional** (`backtester.py:485`); `MonteCarloSimulator` compounds `1+pnl_pct` as equity (`:889-908`) → with 4× leverage the simulated drawdown/ruin is understated ~4×. `np.random.choice` unseeded.
- `max_drawdown` divides by `peak` with no guard (`:251`) while sibling helpers add `1e-10` — inconsistent epsilon hacks.

---

## MEDIUM

- **F18 Session stats double-count trades.** `session.total_trades += 1` on execution (`core/auto_trader.py:583`) and again on close via `update_with_trade_result` (`:952 → :114`); `session.max_drawdown` is measured from start balance, not peak (`:957`). FACT, bug.
- **F19 Duplicate synchronisers/order managers.** `AutoTrader` and every `LiveTradingEnvironment` each own a `PositionSynchronizer` and `OrderManager` (`auto_trader.py:210-221`, `live_trading_env.py:106-117`); stats diverge and callbacks fire twice. FACT, design smell.
- **F20 Feature failures become silent zeros.** `FeatureRegistry.compute_all` catches any exception, logs a warning and writes NaN (`core/feature_registry.py:309-314`); downstream `nan_to_num` turns it into 0 inside model inputs. FACT, design flaw.
- **F21 Live observation padding hides dimension bugs.** `LiveTradingEnvironment._get_observation` pads/truncates to `observation_space` (`core/live_trading_env.py:657-662`). FACT.
- **F22 `load_models` overwrites the running config with the saved `config.json`, including `base_dir` and `mlflow.tracking_uri` absolute paths** (`cli/system.py:1123-1125`, `config/settings.py:183-185, 267`). FACT, portability bug.
- **F23 PaperBroker details.** JPY P&L "converted" by `*0.01` (`core/paper_broker.py:681-683, 746-748`) — not a currency conversion; SL/TP closes at current tick, ignoring the trigger level it computed (`:855-858`); `get_current_tick` comment promises slippage simulation but returns the tick unchanged (`:293-295`); no margin-call/stop-out. FACT.
- **F24 MT5 result semantics.** `OrderResult.success = retcode == 10009` (`core/mt5_broker.py:199`); `10008 PLACED` and `10010 DONE_PARTIAL` are treated as non-retryable failures (`:939-959`) while a position may exist → OrderManager records failure, PositionSynchronizer later sees the position as ours. INFERENCE on frequency; FACT on code. Retry on `result is None` re-sends a market order (`:921-926`) — potential double fill if the first was actually accepted. INFERENCE.
- **F25 `StrategySignal.to_backtest_dict` and `CallableStrategyAdapter` default SL/TP to 50/100 pips regardless of symbol** (`core/strategy.py:59-60, 725-726`); `_get_feature_columns` excludes `time/datetime` but not `timestamp` (`:309`). FACT, brittle.
- **F26 `AutoTraderConfig.initial_balance` does not exist** but is read with `getattr(..., 10000.0)` (`cli/commands/autotrade.py:236`) — config value silently ignored. FACT.
- **F27 Backtester appends final equity twice at the same timestamp** (`backtester.py:308, 320-321`) → spurious zero return; `_calculate_results` mixes `MetricsCalculator` with hand-rolled trade stats. FACT, minor.
- **F28 `AutoTrader._generate_signal` legacy fallback** (`core/auto_trader.py:645-651, 653-782`) duplicates `_combine_signals` logic with a *different* observation source (`env._get_observation()`, z-scored, dim from `model_n_features`) — two code paths that should match but diverge. FACT.

---

## LOW / dead or placeholder code (FACT unless noted; counts exclude tests and `__all__`)

- `core/data_pipeline.py`: `StreamingDataBuffer` (0 refs), `get_online_batch` (0), `inverse_transform_predictions` (0, and a no-op).
- `training/trainer.py`: `train_combined` (0), `_train_curriculum_stage` (only from `train_combined`; contains `if 'data_filter' in stage_config: pass` placeholder at `:370-372`), `evaluate`, `load_all`, `get_training_summary`.
- `training/online_interface.py`: whole module (`OnlineLearningAdapter`, `create_online_learner`) — not used by any CLI or core path.
- `training/online_learning.py:AdaptiveTrainer` — used only by `cli/commands/adapt.py` (which is broken per F9).
- `core/risk_manager.py`: `update_balance`, `record_trade`, `calculate_stop_loss`, `calculate_take_profit`, `update_market_conditions` (all 0 runtime callers).
- `core/position_sync.py`: `PositionTracker` (class defined, no runtime users), `mark_position_as_ours`, `get_change_history`.
- `models/factory.py`: `load_predictor`/`load_agent` (not used by `cli/system.py.load_models`, which reconstructs from `model_metadata.json` instead); `models/base.py` Protocols are type hints only.
- `evaluation/metrics.py`: `information_ratio` only reachable with a benchmark that no caller supplies; `_html_report` unreachable from CLI.
- `utils/pnl_calculator.calculate_pnl_percent`, `utils/position_sizing.calculate_position_size_with_limits` — unused.
- `core/trading_env.py:MultiSymbolTradingEnv` — exported, never constructed.
- `models/transformer.get_attention_weights`, `models/ppo_agent.get_policy_distribution` — unused.
- `scripts/mlflow_ui.py` — thin launcher; fine.
- `core/data_pipeline._normalize_mt5_columns` constant volume 1000 fallback (`:797-800`) — silent fake volume.

---

## Answers to the specific questions

1. **Incomplete**: F3 (live data refresh never implemented), F9/adapt (online learning never wired correctly), trainer curriculum `pass`, `inverse_transform_predictions` no-op, PaperBroker slippage comment.
2. **Silent fallbacks**: synthetic data when MT5 absent (F10); strategy `except Exception → HOLD` (F4); WF `except → hold` (F7); feature registry `except → NaN → 0` (F20); MLflow tracker swallows everything (`utils/mlflow_tracker.py`, many `except Exception: logger.warning`) — acceptable for telemetry.
3. **Fake in runtime paths**: `DefaultPriceProvider` static prices (F8); synthetic random walk (F10); constant `volume=1000` fallback; paper fills assume full fill at tick ± fixed slippage, no partial fills, no swap.
4. **Dead code**: see LOW list.
5. **Wrong wiring**: F1, F2, F4, F5, F6, F7, F12, F15, F16, F18, F22, F26, F28.
6. **Brittle**: `pip_value=0.0001` and `point*10` assumptions; multiplicative spreads; `bars_per_day=24`; `bar_interval=3600`; `timeframe='1h'` in live env; magic 50/100 pip defaults; `_normalize_ratio` log-scaling to avoid float32 overflow is fine but applied inconsistently between train env and strategy (both use it — OK).
7. **Metrics/backtester**: Sharpe/Sortino/Calmar formulas are algebraically standard, but (a) inputs are corrupted by F6, (b) `calculate_all` is stateful (F17), (c) Sortino inconsistent, (d) Monte-Carlo on notional returns, (e) costs: commission+spread+slippage yes, swap no; mark-to-market uses close (fine); same-bar fill (F11).
8. **Concurrency/integration**: PaperBroker locking is consistent (no re-entrancy issues found); AutoTrader thread vs. main-thread `stop()/get_status()` both hit the broker — PaperBroker is locked, MT5 python API is not documented thread-safe (INFERENCE); position/broker state divergence from F5 exposure bookkeeping and F24 partial fills.
9. **Overfitting/leakage**: scaler fit on full data (F1); backtest on training data (F10); WF retrains per fold (yes) but with a mis-shaped model and a different strategy (F7); PPO early stopping on the tail of the same series; predictor early stopping uses val, not test (OK).
10. **Tests**: 321/3; heavy mocking in CLI/strategy tests; no test of any seam listed above.

## Reproduction log (scratch copy, venv)

```
WF train_func X_train shape: (881, 37, 120) -> input_dim = 120 (true n_features=37)
Backtester(RangeIndex df): first timestamps = 2026-09-16 15:00:04.926/.927 (wall clock)
  inferred periods_per_year = 3.3e11 ; sharpe=-1.58e4 ann_return=-1 volatility=1.52e3 avg_dur_h=4.6e-7
  realistic_mode: 12 buy signals over 600 bars -> 5 trades
USDJPY@150: entry cost 1.9 JPY-pips, SL dist 73.1 JPY-pips (asked 50); EURUSD@1.1: SL 55.0 pips (asked 50)
MetricsCalculator same instance hourly->daily ann_return: 317.7 vs 317.7 ; fresh instance daily: 0.2715
Strategy with BrokerPosition: agent.select_action called = False, signal HOLD (agent wanted CLOSE)
  -> AttributeError: 'BrokerPosition' object has no attribute 'direction'
OrderManager+DynamicRiskManager: RM size 92.15 units -> volume 92.15 LOTS ($10.0M notional on $10k) -> "Insufficient margin"
  open notional booked 10000118.0 ; close notional booked 100.00118
autotrade paper path price provider: DefaultPriceProvider
OnlineLearningManager.step x120 (AutoTrader style): agent.experience_buffer = 0 ; mean prediction_error = 44.999 ; online_update X shape (50, 1, 2228)
TradingEnvironment obs z-score: std(z(close)) within window = 3e-6 ; 5219/5220 feature values |z|<0.1
Untrained transformer 'uncertainty' q90-q10 = [0.46 0.55 0.31 0.20] -> confidence 0.45..0.80 vs gate 0.6
```

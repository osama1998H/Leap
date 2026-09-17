# Leap – Predictive / Decision Component Inventory (read-only investigation)

Repo: /home/user/Leap, branch claude/reverse-engineer-redesign-jy7zdo, HEAD aa79be6.
Every claim is tagged FACT (read directly from code, file:line cited) or INFERENCE (conclusion from facts).

## 0. Executive summary

1. Transformer train/inference SCALE MISMATCH (BROKEN). Training scales inputs with a RobustScaler fitted on the whole dataset (core/data_pipeline.py:1006-1011); every inference path (core/strategy.py:_get_prediction ~L424-455, cli/system.py walk-forward wf_strategy ~L1015-1050, core/auto_trader.py:_generate_signal_legacy L672-690, cli/commands/adapt.py:_evaluate_adaptation L368-373) feeds RAW unscaled OHLCV+indicators (prices ~1.1, RSI 0-100, OBV ~1e5). No scaler is saved (utils/checkpoint.py has no scaler field; DataPipeline.inverse_transform_predictions L1110-1121 is a no-op). The #101 fix only guarantees the column COUNT matches. FACT; "production predictions are meaningless" is INFERENCE (very high confidence).

2. PPO observation LAYOUT MISMATCH env vs strategy (BROKEN). Training env: concat(prices[t-w:t].flatten(), features[t-w:t].flatten()) block-wise, z-scored over the whole vector, EXCLUDES current bar (core/trading_env.py:246-258). Strategy (backtest+autotrade): market_data[ohlcv+features].tail(w).values.flatten() row-major/interleaved, NO z-score, INCLUDES current bar (core/strategy.py:_build_observation ~L457-500; evaluation/backtester.py:292-293 passes data.iloc[:i+1]). Dimensions match, semantics don't. FACT. Policy effectively random in production. INFERENCE.

3. Live autotrader trades on a FROZEN snapshot (BROKEN). LiveTradingEnvironment fills buffers once in reset()->_fetch_initial_data() (core/live_trading_env.py:311,559-588; timeframe hard-coded '1h' at L568). Only refresh is _update_data_buffer() inside step() (L362), and AutoTrader never calls env.step() (grep: only online_manager.step at core/auto_trader.py:982). _generate_signal reads env.get_market_data() (L621) -> same stale frame every bar. FACT (grep-verified). _update_data_buffer would also append a tick as a fake bar [bid, ask, bid, bid, 0] (L595-601).

4. Paper mode without MT5 uses a CONSTANT price (core/paper_broker.py:61-72,83-98: EURUSD fixed 1.0850/1.0852). cli/commands/autotrade.py:90-96 builds the paper broker without mt5_broker so DefaultPriceProvider is used despite use_real_prices=True. FACT. Paper P&L can only be spread loss. INFERENCE.

5. Backtest annualised metrics are GARBAGE. LeapTradingSystem.backtest builds the DataFrame with a RangeIndex (cli/system.py:754-770) so Backtester.run uses timestamp=datetime.now() every bar (evaluation/backtester.py:258). PerformanceAnalyzer.analyze passes those to MetricsCalculator.calculate_all -> infer_periods_per_year from microsecond deltas -> Sharpe/vol/annualised return scaled by sqrt(~1e10); trade durations ~0h; daily-trade cap in realistic mode never resets. FACT for the chain; magnitude INFERENCE.

6. Reward design pushes PPO to NEVER TRADE. tc_penalty = 1.0*|dw| (core/trading_env_base.py:305-307; tc_penalty_scale=1.0 core/trading_types.py:155). Default position = 10% of balance (EnvConfig.max_position_size=0.1; sizing core/trading_env.py:299-301) -> |dw|=0.1 -> -0.1 reward on every open AND every close, while log-return of a 10% position on a ~0.1% hourly FX move is ~1e-4. Penalty ~1000x the signal and ~5000x the modelled spread+slippage (2e-5). FACT numbers; "HOLD is the optimum" INFERENCE (high). The zero-trades fix (d411b11) only changed a default confidence constant.

7. Data leakage in the supervised pipeline (moderate). Scaler fit on full series incl. val/test (L1007-1009); ffill().bfill() back-fills warm-up NaNs with future values (L826,L888); MTF features merged with merge_asof(direction='backward') on the higher-TF bar OPEN time (L913-918) so a 4h/1d bar's close-based indicators are visible to 1h bars inside that period (look-ahead up to 3h/23h); train/val split by sample index on overlapping windows (L1058-1074) so last seq_len-1 bars of train windows and h-bar targets of last train samples fall in the val period. FACT mechanics; "MT5 bar time = open time" is domain knowledge -> INFERENCE for impact.

8. NOT a Temporal Fusion Transformer. Vanilla post-LN encoder (+1 extra attention block, +1 GRN before heads), last-timestep readout. VariableSelectionNetwork (L189-277) is defined but never instantiated. No static covariates, LSTM enc/dec, known-future inputs, or multi-horizon output. FACT.

## 1. TRANSFORMER predictor

1.1 Architecture (FACT, models/transformer.py)
- input_projection Linear(input_dim,d_model) L311; input (batch, seq_len, input_dim) L340-351
- sinusoidal PositionalEncoding max_len=max_seq_length L25-47,L314; max_seq_length = config.data.lookback_window (192 in config/templates/data.json; 120 dataclass default config/settings.py:41)
- 4 x TransformerEncoderLayer (custom MHA, post-LN, GELU FFN; d_model=128, n_heads=8, d_ff=512, dropout=0.1) L115-141,L317-320; NO causal mask (fine: whole window is past)
- extra temporal_attention MHA + residual + LN L323,L367-369 ("interpretability")
- pooling x[:, -1, :] L372
- pre_output GatedResidualNetwork(d_model,d_ff,d_model) L326 (only TFT part used)
- heads: point_output Linear(d_model,1) L335; quantile_outputs 3x Linear for q in {0.1,0.5,0.9} L329-332; outputs prediction (B,1), quantiles (B,3,1) L389-393
- unused: VariableSelectionNetwork L189-277

1.2 Inputs / features (FACT)
- prepare_sequences (core/data_pipeline.py:977-1056): features = column_stack([open,high,low,close,volume, data.features]) L996-1003 -> raw prices are inputs.
- data.features = 87 indicators from FeatureRegistry.compute_all (verified by executing the registry: 87 columns incl. sma_*/ema_*/bb_*/keltner_* in price units, obv/vpt/ad cumulative sums up to ~1e5, RSI/MFI/stoch 0-100, hour, day_of_week). input_dim=92 without MTF (matches "(192x87 and 192x128)" in commit 7be4721).
- MTF: up to 25 key features per extra TF appended (L938-975) when additional_timeframes given.
- Scaling: RobustScaler().fit_transform(features) on the ENTIRE array the first time a symbol_timeframe key is seen; later calls transform only (L1006-1011); nan_to_num L1014. Scaler never persisted (TransformerPredictor.save L724-737 saves state dict, optimizer, config, history, input_dim).

1.3 Target (FACT)
- Built in prepare_sequences L1016-1026 (re-implemented in cli/system.py walk-forward train_func ~L925-940 and cli/commands/adapt.py:_prepare_sequences L427-443).
- target='returns': y=(close[t+h]-close[t])/(close[t]+1e-10): simple (not log) forward return over prediction_horizon bars, raw units (~1e-3), unnormalised, single scalar. h = config.data.prediction_horizon (12 default, 6 in data.json). trainer.train_combined L326 / trainer.evaluate L410 call prepare_sequences with library defaults (120,12); the CLI path uses LeapTradingSystem.prepare_training_data (cli/system.py:165-169) which passes config values.
- Alignment: X[i] covers bars i..i+seq-1; y[i]=targets[i+seq-1] = return from bar i+seq-1 to i+seq-1+h (L1041-1051). Correct, no shift bug. n_samples = len-seq-h drops one usable sample (harmless).
- adapt-command variant predicts 1-bar return (adapt.py:441,:374) and trains on market_data.features WITHOUT OHLCV and unscaled (adapt.py:139,155-156) -> input_dim 87 != 92 -> fails at input_projection. INFERENCE from FACTs.

1.4 Leakage: see 0.7. Target computed from raw close, so no target-in-feature leak via the scaled series. FACT.

1.5 Loss/optimiser (FACT)
- L611-622: MSE(point,y) + 0.5*sum_q pinball_q(quantile_q,y).
- AdamW lr=1e-4 wd=1e-5 L453-457; ReduceLROnPlateau(min, factor .5, patience 5) on val loss L459-461,L551.
- epochs=100, batch=64 (shuffled DataLoader L508), grad clip 1.0 L544, early stop patience 15 on val loss, best state restored L553-573,L602-603. Plumbed via cli/system.py:initialize_models L187-202 / training/trainer.py:103-112.

1.6 Evaluation
- Training: only composite train/val loss (trainer.py:96-101).
- ModelTrainer.evaluate (trainer.py:399-418) and cli/commands/evaluate.py:563-565: MAE on raw simple-return scale vs splits['test'] (chronological tail L1058-1074) but evaluate.py re-runs prepare_sequences on a fresh pipeline so the scaler is re-fit on eval data (different fit from training). FACT. Not a true held-out evaluation. INFERENCE.
- adapt --mode evaluate: MAE/RMSE/corr/direction accuracy (adapt.py:387-394) on unscaled OHLCV-less inputs -> invalid.
- No directional-accuracy metric in the main train/backtest path. FACT.

1.7 #101 / dimension fixes
- 7be4721 added the transpose after sliding_window_view (data_pipeline.py:1041-1045, correct), _get_feature_columns to prepend OHLCV (strategy.py:290-331), input_dim check (~L440-449). model_metadata.json stores input_dim + feature_names (cli/system.py:save_models ~L1067-1083); load_models rebuilds network with saved input_dim (~L1141-1160).
- Guaranteed: column count and order. FACT.
- Not guaranteed: (a) value scale; (b) live get_market_data() names columns feature_0..n (live_trading_env.py:245-256) while strategy is created with feature_names=None (auto_trader.py:202) -> auto-detect; count matches only if live pipeline computed same MTF set (autotrade's _fetch_initial_data passes no additional_timeframes -> MTF-trained model would raise); (c) max_seq_length rebuilt from metadata window_size (~L1148) OK.

## 2. PPO agent

2.1 Network (FACT, models/ppo_agent.py)
- ActorCritic L139-280: shared MLP state_dim->256->256 (Linear,ReLU,LayerNorm) L166-170,191-209; actor 256->128->action_dim with tanh(logits)*10 L173-178,L229; critic 256->128->1 L181-186; orthogonal init L211-215; Categorical, deterministic=argmax L250-253.
- PPOAgent L405: gamma .99, gae_lambda .95, clip .2, entropy .01, value .5, max_grad_norm .5, lr 3e-4 Adam eps1e-5, n_steps 2048, n_epochs 10, batch 64 L431-462. CosineAnnealingLR(T_max=100000//2048=48, eta_min=.1 lr) stepped per update L471-477,L732-734 (tied to default 100k, not total_timesteps).
- GAE L558-590, advantage norm L619-628, clipped surrogate + MSE value + entropy L669-689, log-ratio clamp +-20 L670.

2.2 Observation (FACT)
- TradingEnvironment: obs_dim = window*(5+n_features)+8 L104-107; window = config.data.lookback_window (cli/system.py:246) -> 192*92+8 = 17,672 (120*92+8=11048 matches the #101 error).
- Market part L246-258: price_window=data[t-w:t].flatten() (raw OHLCV, excludes bar t), feature_window=features[t-w:t].flatten(), concatenated block-wise, then z-scored over the entire vector with one mean/std.
- Account part core/trading_env_base.py:431-449 (8 values): log1p-scaled balance ratio, equity ratio, n_positions, has_long, has_short, log1p unrealised PnL ratio, max_drawdown, log1p total PnL ratio (overflow fix 4862ce1; _normalize_ratio L414-429).
- Transformer outputs are NOT part of the PPO observation. FACT. Only fusion is rule-based _combine_signals at inference.

2.3 Actions/execution (FACT)
- Discrete(4): HOLD=0 BUY=1 SELL=2 CLOSE=3 (core/trading_types.py:12-17; trading_env_base.py:129). _execute_action L250-264: BUY opens long only if none exists (long+short hedging possible), CLOSE closes all. Fill at close[t] with spread/2+slippage (trading_env.py:262-274), commission size*price*commission L313. Size = balance*0.1/entry unless RiskManager attached (L288-301; none attached by cli/system.py:create_environment L238-247). SL/TP +-2%/+-4% L267-273, checked vs next bar high/low L440-478. Margin check L303-310.

2.4 Reward (FACT, core/trading_env_base.py:266-343)
  prev_equity<=0 -> r=-1
  log_return = clip(log(E_t/E_{t-1}), -0.5, 0.5)
  tc_penalty = -tc_penalty_scale*|w_t-w_{t-1}|, w=clip(sum signed size*entry/equity,-1,1)  (scale 1.0)
  dd_penalty = -dd_penalty_scale*max(0, DD_t-dd_threshold)   (10.0, 0.05)
  vol_penalty = -vol_penalty_scale*Var(last vol_window log_returns)  (0.0 -> off)
  raw = sum; r = clip(raw,-10,10)
  Then RewardNormalizer.normalize (ppo_agent.py:71-90,543): (r-running_mean)/running_std, clip +-10, before storing. SB3 VecNormalize divides by std of discounted returns and does NOT subtract mean; subtracting the mean makes "do nothing" (r=0) negative when running mean>0 -> alters optimal policy. FACT code / INFERENCE consequence.
- Costs enter reward via equity. FACT. tc_penalty ~5000x real modelled cost. INFERENCE.
- Degenerate policies: HOLD forever = 0 raw, no penalty; BUY when already long is a free no-op; opening long+short nets w~0 so second leg is rewarded relative to a single leg (INFERENCE from _get_position_weight netting L357-370).

2.5 Episodes/training (FACT)
- reset(): random start in [window, len-max_episode_steps-1] (trading_env.py:129-135); max_episode_steps=2000 (trading_types.py:148); terminate on equity<=0 or DD>=50% (trading_env_base.py:466-472); end-of-data settlement (trading_env.py:184-204).
- train_on_env (ppo_agent.py:748-945): rollouts of n_steps, update, eval every eval_frequency=10000 (trainer.py:233) on eval_env with 5 deterministic episodes, patience default 15 evals, min_improvement .01, best-state restore. total_timesteps: 1,000,000 dataclass default (settings.py:98) but training.json template = 10,000 -> <5 updates, never reaches first eval. INFERENCE: untrained agent.
- Train/eval env split: last 20% of bars (cli/system.py:280-296); different from transformer's 80/10/10.
- evaluate() sums RAW env rewards L947-964.

2.6 Env features: raw features z-scored per window (env) vs RobustScaler (transformer): two normalisations of the same columns. FACT.

2.7 Online update (FACT, ppo_agent.py:966-1065): random non-sequential samples; 1-step TD target with current critic; PPO ratio vs stale stored log_prob; no EWC/importance weights. INFERENCE: uncorrected off-policy PPO, unstable by construction.

## 3. ONLINE LEARNING (FACT unless noted)
- Trigger: OnlineLearningManager._should_adapt (training/online_learning.py:237-274): step_count % adaptation_frequency(100)==0 AND >=50 prediction errors AND < max_adaptations_per_day(10) AND (mean|pred-actual| > error_threshold 0.05 OR max drawdown of cumulative trading_returns > 0.1). error_threshold 0.05 vs hourly-return target ~1e-3 essentially never fires (INFERENCE); drawdown uses (peak-cum)/(peak+1e-10), unstable when peak~0.
- Regime detection: MarketRegimeDetector.detect_regime L67-94: needs >=25 prices; vol=mean|returns| over 50; high_volatility if vol>0.02; trending_up if (p_last-p_0)/p_0>0.02 and SMA10>SMA50; trending_down symmetric; low_volatility if vol<0.005; else ranging. Regime string only logged; nothing consumes it. DynamicRiskManager.update_market_conditions (risk_manager.py:480-492) is a separate heuristic NEVER called (grep).
- Retrained: _adapt_predictor L310-332: last 50 (features, actual), reshaped (N,1,F) i.e. seq_len=1, -> TransformerPredictor.online_update (single grad step lr 1e-5, transformer.py:686-722). _adapt_agent L334-346 -> PPOAgent.online_update.
- Catastrophic-forgetting prevention: none (small LR + grad clip only; no EWC, replay of old data, validation gate, rollback).
- Wired into autotrade? Partially and inertly: autotrade.py:63-68 creates the manager; AutoTrader._on_position_closed (auto_trader.py:961-989) calls online_manager.step WITHOUT log_prob/value so agent.store_transition is skipped (online_learning.py:172-181) -> PPO buffer never fills -> _adapt_agent never runs. market_data['features'] is set to the flattened PPO observation (17k-vector) L974-976 -> _adapt_predictor would build X (N,1,17672) -> input_projection dim error (caught by logger.exception). _check_adaptation L858-891 triggers after 100 executed TRADES per day (not bars). Net: inert/erroring. INFERENCE from FACTs.
- adapt command (cli/commands/adapt.py): offline = full predictor.train + agent.train_on_env on recent data via AdaptiveTrainer.train_offline (L197-202) = plain fine-tuning; uses features without OHLCV and unscaled -> shape error vs 92-dim model. online mode streams historical bars with reward=0.0 always (L302), predict(features.reshape(1,1,-1)) seq_len=1. training/online_interface.py OnlineLearningAdapter exported but used by nothing (grep).

## 4. HEURISTICS / RULES (deterministic, not learned)

4.1 Signal fusion CombinedPredictorAgentStrategy._combine_signals (core/strategy.py ~L590-650) FACT
  confidence = clip(1-(q90-q10),0,1)   # spread in raw-return units
  if confidence < min_confidence(0.6): HOLD
  if not open_status: CLOSE if agent==CLOSE else HOLD
  if agent==CLOSE: CLOSE
  if agent==BUY : BUY if pred>=+thr; HOLD if pred<-thr; else BUY   (thr=prediction_threshold=0.001)
  if agent==SELL: SELL if pred<=-thr; HOLD if pred>+thr; else SELL
  else HOLD
- Agent has priority; transformer can only veto. Transformer alone never opens a trade (agent None -> HOLD), contradicting the "Transformer only" log at cli/system.py ~L812. FACT.
- Confidence gate is a no-op on a sane model (spread ~0.01 -> conf ~0.99). INFERENCE. uncertainty is a (1,1) ndarray; works by accident.
- Legacy duplicate AutoTrader._combine_signals L729-782 (only on strategy exception).

4.2 Position sizing (FACT)
- Training env (no RM): size = balance*0.10/entry (trading_env.py:299-301; utils/position_sizing.py:56-84).
- Backtester (no RM attached by system.backtest): size = balance*0.02/(sl_pips*0.0001*entry), capped at balance*leverage/entry and optional max_position_size (1e6 realistic) (backtester.py:149-185,408-414; position_sizing.py:12-53,87-132).
- Live OrderManager with DynamicRiskManager: RiskManager.calculate_position_size L189-226, method='kelly' default (L33): wins+losses<20 -> balance*0.01/entry; else half-Kelly f=clip(0.5*(p*b-q)/b,0,0.25), size=balance*f/|entry-SL|; then min(size, 2% bal/entry, remaining 10% exposure/entry, bal*10/entry); Dynamic x0.5 high-vol, x1.2 low-vol, x0.7 after >=3 consecutive losses (L494-519). order_manager.py:_calculate_position_params clamps to volume_min/max/step.
- 'percent','volatility','fixed' methods implemented L203-220, never selected.
- Nothing calls RiskManager.update_balance, record_trade, update_market_conditions (grep non-test code). So wins/losses stay 0 -> Kelly branch never reached; current_drawdown never updates -> drawdown/daily/weekly/consecutive-loss halts (_check_risk_limits L375-406) can never trigger; regime stays 'normal'. Only on_position_opened/closed (exposure/count) are wired (trading_env.py:331,388; backtester.py:455,507). FACT.

4.3 Stops/targets (FACT)
- Env: SL 2%/TP 4% of entry, high/low trigger, no trailing (trading_env.py:267-273,440-478).
- Backtester: SL/TP pips from signal (50/100 default via StrategySignal.to_backtest_dict strategy.py:55-61), fill at SL/TP level with spread/slippage applied on exit (backtester.py:459-479,525-555).
- Live: pips->price via point*10 (order_manager.py); broker-managed. RiskConfig.use_trailing_stop/trailing_stop_pips (settings.py:118-119) never read. RiskManager.calculate_stop_loss/take_profit (ATR) L268-312 never called.
- should_take_trade (RR>=1.5, direction sanity, max positions) called only when RM attached; backtest path has none.

4.4 Other gates
- Backtester realistic mode: >=4 bars between trades, <=5 trades/day, <=1e6 units (backtester.py:121-125,340-349); daily counter needs real timestamps (broken in system.backtest).
- OrderManager._validate_signal (order_manager.py:357-398): confidence>=0.5, connected, symbol exists, spread<=max_spread_pips, account trade_allowed, RM allowed & positions<max.
- AutoTrader: trading hours/days (L809-827), daily-loss stop 5% vs _daily_start_balance (L829-856), new-bar detection = wall-clock elapsed >= bar_interval (L784-807), not broker bar time.

## 5. Component table (Model | Implementation | Library | Purpose | Inputs | Target | Training data | Output | Metric | Production use | Status)

- Price predictor | vanilla transformer encoder 4x d=128 8 heads + GRN + point&3 quantile heads; MSE+0.5 pinball; AdamW 1e-4; ReduceLROnPlateau; early stop 15 | models/transformer.py TemporalFusionTransformer L279-405, TransformerPredictor L408-762; training/trainer.py:train_predictor L72-162 | PyTorch | forecast h-bar forward simple return | (B,lookback,5+87[+MTF]) OHLCV+indicators, RobustScaler (train)/raw (inference) | (close[t+h]-close[t])/close[t] raw | fetch_historical_data -> prepare_sequences -> 80/10/10 index split (cli/system.py:165-178) | point return + q10/q50/q90 | train/val composite loss; test MAE (trainer.py:413, evaluate.py:564) | loaded by load_models (system.py ~L1120-1165); invoked in Backtester.run via strategy._get_prediction, AutoTrader._generate_signal, walk-forward wf_strategy, adapt | BROKEN in production: scale mismatch; can only veto; leaky fit; not a TFT
- RL policy | PPO discrete-4 actor-critic MLP [256,256]+[128] heads, GAE, clip .2, cosine LR, running reward normaliser | models/ppo_agent.py PPOAgent L405-1117; core/trading_env.py, core/trading_env_base.py; trainer.py:train_agent L164-290 | PyTorch, gymnasium | choose HOLD/BUY/SELL/CLOSE | flat z-scored [OHLCV window || feature window] (excl current bar) + 8 account feats (17,672 @192x92) | max sum normalised v2 reward | same MarketData; first 80% train env, last 20% eval env (system.py:280-296); random 2000-step episodes | action logits/argmax | raw episode reward, eval reward, policy/value loss, entropy, clip frac, action dist (MLflow) | loaded by load_models; agent.select_action(obs, deterministic=True) from strategy._build_observation in backtest & autotrade | BROKEN in production: obs layout/normalisation/off-by-one mismatch; reward dominated by tc_penalty -> HOLD optimum; template total_timesteps=10000; reward mean subtraction
- Signal fusion | rule table | core/strategy.py CombinedPredictorAgentStrategy ~L246-655 | - | combine models | pred return, quantile spread, agent action, open_status | - | - | StrategySignal (SL 50/TP 100 pips, risk 2%) | - | Backtester.run L292-304; AutoTrader._generate_signal L635-643 | WORKING as written but confidence gate vacuous; transformer-only mode always HOLDs
- Risk manager | limits + Kelly/percent/vol sizing + regime scaling | core/risk_manager.py RiskManager L53-467, DynamicRiskManager L470-519 | numpy | sizing, halts, RR gate | balance, W/L stats, exposure | - | - | size, allow/deny | - | DynamicRiskManager(initial_balance=backtest.initial_balance) in system.risk_manager L109-115; passed to AutoTrader/OrderManager/LiveTradingEnvironment only; NOT to training env or backtester | QUESTIONABLE/mostly inert: no caller of update_balance/record_trade -> halts & Kelly never activate; live sizing = 1% of balance/entry capped at 2% notional
- Regime detector | thresholds on 50-bar trend & mean|return| | training/online_learning.py MarketRegimeDetector L51-94 | numpy | label regime | close, 1-bar return | - | - | string | - | updated in OnlineLearningManager.step; consumed by nothing | UNUSED (decorative)
- Online adaptation manager | error/DD-triggered single-step fine-tune | OnlineLearningManager L97-407 | - | adapt live | pred error, PnL, buffers | - | last 50 samples | loss | mean pred error, simplified Sharpe | wired in autotrade but fed malformed data | BROKEN/inert
- AdaptiveTrainer | offline retrain wrapper + threaded stream loop | online_learning.py L410-576; cli/commands/adapt.py | - | adapt CLI | features (no OHLCV, unscaled) | 1-bar return | recent n_bars | - | MAE/RMSE/corr/dir-acc (evaluate mode) | adapt command only | BROKEN for a normally-trained model (input_dim 87 vs 92, scale)
- OnlineLearningAdapter | protocol shim | training/online_interface.py | - | unify online_update | - | - | - | - | - | none | UNUSED
- Backtester | event loop, SL/TP on high/low, spread+slippage+commission, risk sizing, cooldown/daily caps | evaluation/backtester.py Backtester L76-676 | pandas | evaluate strategy | OHLCV+features DataFrame | - | - | BacktestResult | Sharpe, Sortino, Calmar, DD, PF, VaR... via MetricsCalculator | system.backtest (backtest CLI) | QUESTIONABLE: annualised from datetime.now() timestamps; O(n^2) data.iloc[:i+1] per bar
- Walk-forward | 180/30-day rolling folds (24 bars/day hard-coded), fresh transformer per fold, +-0.1% rule | backtester.py WalkForwardOptimizer L679-863; cli/system.py:walk_forward_test ~L853-1060 | - | OOS validation | raw OHLCV+features | h-bar return | fold train window (80/20 in-fold) | aggregated fold stats | mean/std return, Sharpe, DD, win-rate, profitable ratio | walkforward CLI | QUESTIONABLE: internally consistent (both raw) but unscaled inputs into a transformer; PPO not evaluated; same timestamp issue
- Monte Carlo | bootstrap of trade pnl_pct | backtester.py MonteCarloSimulator L866-938 | numpy | risk of ruin | closed trades | - | - | percentiles | - | PerformanceAnalyzer with --monte-carlo | working
- Live env | broker-synced gym env | core/live_trading_env.py | - | live obs, execute via OrderManager | broker ticks, buffered bars | - | - | obs | - | AutoTrader._initialize_environments L449-468 | BROKEN: buffer never refreshed after reset; hard-coded '1h'; tick->bar hack
- Paper broker | static price table | core/paper_broker.py DefaultPriceProvider L55-119 | - | paper trading | - | - | - | constant tick | - | autotrade --paper (no mt5_broker) | BROKEN as a P&L test

## 6. Checkpoint / loading consistency (FACT)
- utils/checkpoint.py:save_checkpoint L184-211 -> {model_state_dict, optimizer_state_dict, config, training_history, metadata{model_type,input_dim|state_dim,action_dim}}; legacy mapping on load L214-286.
- trainer._save_predictor_checkpoint/_save_agent_checkpoint (trainer.py:430-468) write checkpoints/predictor_<ts>.pt + _info.json; NOTHING loads these (grep). trainer.save_all writes predictor.pt/agent.pt WITHOUT model_metadata.json (trainer.py:470-480); LeapTradingSystem.train calls save_all(save_dir) L379 then CLI calls system.save_models(args.model_dir) (train.py:340) which writes metadata; if args.model_dir != config.models_dir, .pt files and metadata land in different dirs.
- load_models refuses to load without model_metadata.json (~L1106-1109); rebuilds predictor with input_dim and max_seq_length=window_size from metadata, agent with state_dim/action_dim; dims consistent with what was saved.
- Scalers never saved; DataPipeline.scalers in-memory only. Even a correct inference path could not reproduce training scaling.
- feature_names saved are computed names; live get_market_data() uses feature_0.. names, which would be filtered out if feature_names were passed (it isn't: auto_trader.py:202 passes None).
- Hyper-params at load come from saved config.json (SystemConfig.load ~L1100); checkpoint config overwrites self.config inside TransformerPredictor.load L750-751 after construction (architecture cannot change; LR etc. can).
- Optimizer state restored on load (L747-748; ppo_agent.py:1095-1096); PPOAgent.reward_normalizer stats NOT saved.

## 7. Specific questions
- Transformer output at t from data <= t? Backtest yes (data.iloc[:i+1].tail(w)). Training scaler and MTF features may embed future (0.7). FACT/INFERENCE.
- PPO obs includes transformer outputs? No. FACT.
- Costs in reward? Yes via equity (spread/2+slippage entry & manual exit, commission both ways; SL/TP fills commission only). Synthetic tc_penalty ~5000x larger. FACT/INFERENCE.
- Overflow fix 4862ce1: log1p on ratios (trading_env_base.py:414-449) mirrored in strategy._build_account_observation; consistent. Market z-scoring NOT mirrored.
- Zero-trades fix d411b11: default prediction_confidence 0.5->0.7 so the 0.6 gate passed when predictor threw. Deeper causes remain (reward, obs mismatch, transformer-only path never trades).
- Same scaled features in env as training? Env z-scores per window in both TradingEnvironment and LiveTradingEnvironment (live_trading_env.py:377-407 DOES z-score, unlike strategy._build_observation). The legacy autotrader path (only on exception) is closer to training than the primary strategy path.

## 8. Smaller findings (FACT unless noted)
- trainer.train_combined/_train_curriculum_stage use defaults (120,12) and split .7/.15; curriculum data_filter is a pass (L370-372).
- TransformerPredictor.predict always return_attention=True (L667): wasted memory (4 layers of (B,8,192,192)).
- RewardNormalizer._update variance recursion (L119) is non-standard; var init 1.0 then 0.0 at n=1; early normalised rewards wildly scaled. INFERENCE.
- CosineAnnealingLR T_max=48 updates regardless of total_timesteps; with 1M steps LR cycles ~10x. INFERENCE.
- WalkForwardOptimizer.generate_splits hard-codes 24 bars/day (L707).
- Backtester.run passes data.iloc[:i+1] per bar -> O(n^2) copies + per-bar 192-step transformer forward; very slow at 50k bars. INFERENCE.
- AutoTrader._is_new_bar uses wall-clock since process start.
- training.json ppo.total_timesteps=10000 vs 1,000,000 default -> silently untrained agent.
- RiskConfig.use_trailing_stop, trailing_stop_pips, risk_reward_ratio, EvaluationConfig.metrics never read.

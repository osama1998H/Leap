# Data agent — condensed (full text in conversation; scripts t1..t7 + .out in scratchpad)
CRITICAL
1. Silent synthetic fallback: data_pipeline.py:618,654-668 -> random walk whenever not (MT5_AVAILABLE and is_connected). On Linux/macOS every command runs on synthetic GBM (verified t4). system.load_data never checks is_connected. --save-data metadata data_source reads non-existent broker_gateway attr -> always "synthetic".
2. Multi-timeframe leak: merge_asof(direction='backward') on bar-OPEN timestamps (data_pipeline.py:913-918) -> 1h bars 00:00-02:00 receive the 4h bar whose close is 03:59. Verified t2 (perturb 03:00 bar -> 24 4h_* features change on 3 preceding rows). Only with --multi-timeframe.
3. Scaler: RobustScaler fit on ALL bars incl test (1005-1011); never saved; never applied at inference. t3: perturbing last 20% of bars changes X[0]. t7: train |max| 4.5 vs inference |max| 33,016.
4. PPO obs mismatch: env z-scored excl. current bar vs strategy raw incl. current bar. t7: env std 1.0 vs strategy std 1922.
5. Live frozen snapshot: env.reset once -> _fetch_initial_data('1h', window+50) ; step never called. get_market_data renames features feature_0..N and DROPS them when buffers diverge (t5: after one step -> 5 columns -> "model expects 92, has 5" -> HOLD). _update_data_buffer fabricates bars [bid, ask, bid, bid, 0] without features. _resolve_feature_dim defaults to 100 because DataPipeline.feature_count doesn't exist.
HIGH
6. ffill().bfill() (826, 888, 930-931) back-fills first 199 rows of sma_200 etc. with bar 199's value -> future info in warm-up rows used as training samples. Live with 110 bars: sma_200 all NaN -> 0.
7. No purge/embargo at split boundaries: 12-bar target overlap, 119-bar input overlap (t3). Same for PPO 80/20 and WFO 80/20.
8. Backtester RangeIndex -> datetime.now() per bar -> periods/yr 5.18e11 (t6) -> Sharpe/annualised meaningless; daily trade cap never resets.
9. Decision/fill timing inconsistent: RL train decides on <=t-1 fills close[t]; backtest decides on <=t fills close[t]; live fills at next tick after wall-clock boundary.
10. Raw non-stationary inputs: OHLCV levels, price-unit indicators, obv/vpt/ad cumsums from first fetched bar (depend on n_bars).
MEDIUM
11. Timezone: naive timestamps; broker time vs UTC trading hours.
12. No missing-bar/weekend/duplicate/forming-bar handling. Synthetic includes weekends (576/2000), MT5 doesn't -> hour/dow distributions differ.
13. Synthetic: sigma 1%/bar (~78%/yr vs ~8% real EURUSD), seed hash(timeframe) non-deterministic across processes, reseeds GLOBAL numpy RNG.
14. adapt/online-learning use different inputs & targets than training (87 vs 92 dims; 1-bar horizon; flattened PPO obs fed to transformer; trade PnL as target).
15. volatility_* uses sqrt(252) on hourly; MetricsCalculator periods_per_year=252 in envs.
16. WFO fixed 24 bars/day.
18. No class balance handling; direction target unused; regression target not winsorised.
WHAT IS CORRECT: all 87 registry features are trailing/causal (t1); transformer target alignment correct (t3); RL env step timing correct (act close[t], MTM t+1); splits chronological; backtester applies spread/slippage/commission and SL/TP vs high/low.
Data sources table: MT5 copy_rates_from(symbol, tf, datetime.now() naive, n_bars=50000 default) -> columns time,O,H,L,C,tick_volume,spread,real_volume -> spread discarded; bars bid-based; forming bar included (inference). Synthetic: N(1e-4, 1e-2) returns, 1.1*exp(cumsum). Paper broker: static EURUSD 1.0850/1.0852 + synthetic ticks. NO CSV loader, NO yfinance.
Features: 87 registered (returns, log_returns, hl/oc ratio, gap, tr, sma/ema 5-200, ratios, crosses, rsi 7/14/21 + wilder, macd, stoch, williams, roc, momentum, atr, bollinger, keltner, volatility 10/20/30, volume sma/ratio, obv/vpt/ad, mfi, trend, cci, ADX family (10), candle patterns (8), time features (6)). +24 per extra TF via hardcoded key list.

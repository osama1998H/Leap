import sys, logging; sys.path.insert(0, '/home/user/Leap')
logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd
from core.data_pipeline import DataPipeline
from core.trading_env import TradingEnvironment
from core.strategy import CombinedPredictorAgentStrategy, StrategyConfig
rng = np.random.default_rng(3)
# ---- (1) bfill head leak: perturb bar 199 only, check rows 0..198
N=600; close = 1.1*np.exp(np.cumsum(rng.normal(0,0.001,N))); ts = pd.date_range('2024-01-01', periods=N, freq='1h')
df = pd.DataFrame({'timestamp': ts,'open': close,'high': close*1.0005,'low': close*0.9995,'close': close,'volume': rng.uniform(1e3,1e4,N)})
dp = DataPipeline(); md = dp._process_raw_data(df.copy(),'X','1h')
df2 = df.copy(); df2.loc[199, ['open','high','low','close']] *= 1.02
md2 = dp._process_raw_data(df2.copy(),'X','1h')
F1 = pd.DataFrame(md.features, columns=md.feature_names); F2 = pd.DataFrame(md2.features, columns=md2.feature_names)
chg = {c: int((~np.isclose(F1[c].iloc[:199], F2[c].iloc[:199])).sum()) for c in md.feature_names if (~np.isclose(F1[c].iloc[:199], F2[c].iloc[:199])).any()}
print("(1) bfill leak: perturb ONLY bar 199 -> #rows in [0,198] changed per feature:", chg)
print("    sma_200 rows 0..198 all identical to row 199 value?", bool(np.all(F1['sma_200'].iloc[:199].to_numpy()==F1['sma_200'].iloc[199])))
# ---- (2) PPO observation: training env vs strategy inference on the SAME bars
W=60
ohlcv = np.column_stack([md.open, md.high, md.low, md.close, md.volume])
env = TradingEnvironment(data=ohlcv, features=md.features, window_size=W)
env.current_step = 400
obs_env = env._get_market_observation()
dfi = pd.DataFrame(ohlcv, columns=['open','high','low','close','volume']); 
for i,n in enumerate(md.feature_names): dfi[n] = md.features[:,i]
strat = CombinedPredictorAgentStrategy(predictor=None, agent=None, config=StrategyConfig(lookback_window=W), feature_names=list(md.feature_names))
obs_str = strat._build_observation(dfi.iloc[:401], positions=[], account_state=None)[:-8]  # market part only, same "current" bar 400
print(f"(2) PPO market obs @bar400 | training env: mean={obs_env.mean():.3f} std={obs_env.std():.3f} min={obs_env.min():.1f} max={obs_env.max():.1f} (window bars [340,399], z-scored jointly)")
print(f"                              | strategy inference: mean={obs_str.mean():.1f} std={obs_str.std():.1f} min={obs_str.min():.1f} max={obs_str.max():.1f} (window bars [341,400], RAW, includes current bar)")
# ---- (3) Transformer input: training (RobustScaler) vs backtest/live inference (raw)
L=120; H=12
X,y,_ = dp.prepare_sequences(md, L, H)
print(f"(3) Transformer X train: per-feature median≈0, IQR≈1 -> overall mean={X.mean():.3f} std={X.std():.2f}, |max|={np.abs(X).max():.1f}")
cols = ['open','high','low','close','volume']+list(md.feature_names)
Xinf = dfi[cols].iloc[400-L+1:401].values[None]
print(f"    Transformer X inference (strategy._get_prediction, no scaler): mean={Xinf.mean():.1f} std={Xinf.std():.1f}, |max|={np.abs(Xinf).max():.1f}; e.g. volume col mean={Xinf[0,:,4].mean():.0f}, obv col mean={Xinf[0,:,cols.index('obv')].mean():.0f}")
# ---- (4) live history length vs indicator warm-up
for nb in (60+50, 192+50):
    m = dp.fetch_historical_data('EURUSD','1h', n_bars=nb)
    F = pd.DataFrame(m.features, columns=m.feature_names)
    allnan = [c for c in m.feature_names if F[c].isna().all()]
    print(f"(4) live _fetch_initial_data n_bars={nb}: all-NaN features after ffill/bfill = {allnan}; rows where sma_200 is bfilled copy = {int((F['sma_200']==F['sma_200'].iloc[0]).sum()) if 'sma_200' in F and not F['sma_200'].isna().all() else 'n/a'}")
print("(4b) DataPipeline has attribute 'feature_count'?", hasattr(dp,'feature_count'), "| 'broker_gateway'?", hasattr(dp,'broker_gateway'))

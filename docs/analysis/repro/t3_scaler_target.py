import sys, logging; sys.path.insert(0, '/home/user/Leap')
logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd
from core.data_pipeline import DataPipeline, MarketData
rng = np.random.default_rng(2)
N=3000; L=120; H=12
close = 1.1*np.exp(np.cumsum(rng.normal(0,0.001,N)))
ts = pd.date_range('2024-01-01', periods=N, freq='1h')
df = pd.DataFrame({'timestamp': ts,'open': close,'high': close*1.0005,'low': close*0.9995,'close': close,'volume': rng.uniform(1e3,1e4,N)})
dp = DataPipeline(); md = dp._process_raw_data(df.copy(),'EURUSD','1h')
X,y,t = dp.prepare_sequences(md, L, H)
print("X,y shapes:", X.shape, y.shape, "| n_bars:", N, "| n_samples = N-L-H =", N-L-H)
# Target alignment check
i = 777
exp = (close[i+L-1+H]-close[i+L-1])/close[i+L-1]
print(f"y[{i}] == (close[i+L-1+H]-close[i+L-1])/close[i+L-1]? ", np.isclose(y[i], exp), "| timestamp of sample = bar index", i+L-1, "->", t[i]==md.timestamp[i+L-1])
# Does the last input step contain the target? check scaled close feature (col 3) at last step vs unscaled
sc = dp.scalers['EURUSD_1h']
raw_last_close = X[i,-1,3]*sc.scale_[3]+sc.center_[3]
print("last-step close in X[i] recovers close[i+L-1]:", np.isclose(raw_last_close, close[i+L-1]), "(not close[i+L-1+H]):", not np.isclose(raw_last_close, close[i+L-1+H]))
# Scaler fit-on-full-data leak: perturb ONLY the final 20% of bars, check X[0] changes
dp2 = DataPipeline(); df2 = df.copy(); cut=int(N*0.8)
df2.loc[cut:, ['open','high','low','close']] *= 1.5; df2.loc[cut:,'volume'] *= 10
md2 = dp2._process_raw_data(df2.copy(),'EURUSD','1h'); X2,_,_ = dp2.prepare_sequences(md2, L, H)
print("Perturb only last 20% of bars -> X[0] (earliest training sample) changed:", not np.allclose(X[0], X2[0]),
      "| max abs diff in X[0]:", float(np.abs(X[0]-X2[0]).max()))
print("RobustScaler center_ fit on ALL", sc.n_features_in_, "columns incl. raw OHLCV; center_[3] (close) =", round(float(sc.center_[3]),5), "median(close all bars)=", round(float(np.median(close)),5))
# Split: is the train/val boundary purged?
sp = dp.create_train_val_test_split(X,y,0.8,0.1)
ntr = len(sp['train'][0])
print("train/val/test sizes:", [len(sp[k][0]) for k in ('train','val','test')])
print(f"Last train sample idx {ntr-1}: input covers bars [{ntr-1},{ntr-1+L-1}], target uses close up to bar {ntr-1+L-1+H}")
print(f"First val sample idx {ntr}: input covers bars [{ntr},{ntr+L-1}] -> overlap of train TARGET horizon with val INPUT bars: {H} bars; input overlap: {L-1} bars (no purge/embargo)")
# Direction class balance on random walk
d = (y>0).mean(); print("direction target balance P(up) on synthetic:", round(float(d),3))
# Does system.py env split leak? train env = bars[:split], eval env = bars[split:], features from full-data pass (bfill only affects head)
print("Note: features passed to TradingEnvironment are md.features (UNscaled); prepare_sequences scaler is never applied to env or to inference.")

import logging, numpy as np, pandas as pd, warnings
logging.disable(logging.CRITICAL); warnings.filterwarnings("ignore")
print("=== 8) OnlineLearningManager.step as called by AutoTrader._on_position_closed ===")
from training.online_learning import OnlineLearningManager
class A:
    experience_buffer=[]
    def store_transition(self,**k): A.experience_buffer.append(k)
class P:
    calls=[]
    def online_update(self,X,y): P.calls.append(X.shape); return 0.0
olm=OnlineLearningManager(predictor=P(), agent=A())
obs=np.zeros(60*(5+32)+8, dtype=np.float32)   # flattened PPO observation, as AutoTrader passes
for _ in range(120):
    r=olm.step({'state':obs,'features':obs,'close':1.1,'returns':0.0}, prediction=0.001, actual=45.0, trading_action=1, trading_reward=50.0, done=True)
print("agent.experience_buffer after 120 closed trades:", len(A.experience_buffer))
print("avg 'prediction_error' (|0.001 - profit/entry_price|):", round(float(np.mean(olm.prediction_errors)),3), "> error_threshold 0.05 -> adaptation forced")
print("predictor.online_update called with X shape:", P.calls[:1], "(transformer expects (batch, seq_len, 5+n_features))")

print("\n=== 9) TradingEnvironment observation normalization: variance share ===")
from core.data_pipeline import DataPipeline
dp=DataPipeline(); md=dp.fetch_historical_data('EURUSD','1h',n_bars=1500)
ohlcv=np.column_stack([md.open,md.high,md.low,md.close,md.volume])
w=60; s0=500
pw=ohlcv[s0-w:s0].flatten(); fw=md.features[s0-w:s0].flatten(); obs=np.concatenate([pw,fw])
mu,sd=obs.mean(),obs.std(); z=(obs-mu)/sd
print("n_features=%d; obs mean=%.1f std=%.1f" % (md.features.shape[1],mu,sd))
print("std of z-scored close within window: %.6f ; std of raw close: %.5f" % (z[:w*5].reshape(w,5)[:,3].std(), ohlcv[s0-w:s0,3].std()))
print("|z|>0.1 among 300 OHLCV values: %d ; among %d feature values: %d" % ((np.abs(z[:w*5])>0.1).sum(), len(fw), (np.abs(z[w*5:])>0.1).sum()))
print("features containing '200':", [f for f in md.feature_names if '200' in f])
raw=dp.feature_engineer._registry.compute_all(pd.DataFrame({'timestamp':md.timestamp,'open':md.open,'high':md.high,'low':md.low,'close':md.close,'volume':md.volume}))[md.feature_names].values
print("NaN fraction in first 200 feature rows before ffill/bfill: %.3f -> filled by bfill from FUTURE rows" % np.isnan(raw[:200]).mean())
md2=dp.fetch_historical_data('EURUSD','1h',n_bars=110)  # live _fetch_initial_data: window_size+50
print("live fetch (110 bars): NaN fraction remaining in features after ffill/bfill: %.3f" % np.isnan(md2.features).mean())

print("\n=== 10) Transformer 'uncertainty' (q90-q10) vs min_confidence=0.6 ===")
from models.transformer import TransformerPredictor
tp_=TransformerPredictor(input_dim=10, config={'d_model':16,'n_heads':2,'n_encoder_layers':1,'d_ff':32,'max_seq_length':20}, device='cpu')
Xr=np.random.randn(4,20,10).astype(np.float32)
r=tp_.predict(Xr, return_uncertainty=True); print("uncertainty:", r['uncertainty'].ravel().round(4), "-> confidence=1-u:", (1-r['uncertainty'].ravel()).round(3))

print("\n=== 11) inverse_transform_predictions is a no-op ===")
import inspect; print(inspect.getsource(dp.inverse_transform_predictions).splitlines()[-5:])

print("\n=== 12) data_source label: DataPipeline has broker_gateway attr? ===", hasattr(dp,'broker_gateway'))

import sys, logging; sys.path.insert(0, '/home/user/Leap')
logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd
from core.data_pipeline import DataPipeline
rng = np.random.default_rng(1)
N = 24*60  # 60 days hourly
ts = pd.date_range('2024-01-01', periods=N, freq='1h')
close = 1.1*np.exp(np.cumsum(rng.normal(0,0.001,N)))
h1 = pd.DataFrame({'timestamp': ts, 'open': close, 'high': close*1.0005, 'low': close*0.9995, 'close': close, 'volume': 1000.0})
# Build 4h bars the way MT5 does: bar time = OPEN time of the bar (label='left')
def to4h(df):
    g = df.set_index('timestamp').resample('4h', label='left', closed='left')
    out = pd.DataFrame({'open': g['open'].first(), 'high': g['high'].max(), 'low': g['low'].min(), 'close': g['close'].last(), 'volume': g['volume'].sum()}).reset_index()
    return out
h4 = to4h(h1)
dp = DataPipeline()
md = dp._process_raw_data(h1.copy(), 'EURUSD', '1h', additional_timeframes_data={'4h': h4})
F = pd.DataFrame(md.features, columns=md.feature_names, index=h1['timestamp'])
# Perturb 1h bar at 2024-02-10 03:00 (the LAST hour inside the 4h bar that opened 00:00)
tpert = pd.Timestamp('2024-02-10 03:00')
h1p = h1.copy(); h1p.loc[h1p['timestamp']==tpert, ['open','high','low','close']] *= 1.03
h4p = to4h(h1p)
mdp = dp._process_raw_data(h1p.copy(), 'EURUSD', '1h', additional_timeframes_data={'4h': h4p})
Fp = pd.DataFrame(mdp.features, columns=mdp.feature_names, index=h1p['timestamp'])
cols4h = [c for c in md.feature_names if c.startswith('4h_')]
print("4h feature columns added:", cols4h)
before = F.index < tpert
chg = {}
for c in cols4h:
    d = ~np.isclose(F[c][before], Fp[c][before], equal_nan=True)
    if d.any():
        idx = F.index[before][d]
        chg[c] = (str(idx.min()), str(idx.max()), int(d.sum()))
print(f"Perturbed only 1h bar at {tpert}. Primary rows STRICTLY BEFORE it whose 4h_* features changed:")
for k,v in chg.items(): print("   ", k, "first/last changed ts, n:", v)
if not chg: print("   NONE")
# Show the direct mechanism: at primary 00:00,01:00,02:00 the 4h_returns equals return of 4h bar closing at 03:59
r4 = h4.set_index('timestamp')['close'].pct_change()
print("4h bar 2024-02-10 00:00 return (uses close @03:00):", round(float(r4.loc['2024-02-10 00:00']),6),
      "| 4h_returns visible to 1h rows 00:00..03:00:", [round(float(F['4h_returns'].loc[f'2024-02-10 0{i}:00']),6) for i in range(4)])

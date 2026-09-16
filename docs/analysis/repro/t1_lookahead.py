import sys, logging; sys.path.insert(0, '/home/user/Leap')
logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd
from core.feature_registry import FeatureRegistry
from core.data_pipeline import DataPipeline

rng = np.random.default_rng(0)
N = 600
ret = rng.normal(0, 0.001, N)
close = 1.1*np.exp(np.cumsum(ret))
ts = pd.date_range('2024-01-01', periods=N, freq='1h')
df = pd.DataFrame({'timestamp': ts, 'open': close*(1+rng.normal(0,2e-4,N)), 'close': close,
                   'volume': rng.uniform(1000,10000,N)})
df['high'] = df[['open','close']].max(axis=1)*(1+rng.uniform(0,1e-3,N))
df['low']  = df[['open','close']].min(axis=1)*(1-rng.uniform(0,1e-3,N))

reg = FeatureRegistry.get_instance()
base = reg.compute_all(df.copy())
feat_cols = [c for c in base.columns if c not in ['timestamp','open','high','low','close','volume']]
print(f"n_features registered/computed: {len(reg)} / {len(feat_cols)}")

# --- Test A: pure registry causality. Perturb bars strictly after K; compare rows <= K.
K = 400
df2 = df.copy()
for c in ['open','high','low','close']: df2.loc[K+1:, c] *= (1 + rng.normal(0, 0.05, N-K-1))
df2.loc[K+1:, 'volume'] *= 3
pert = reg.compute_all(df2)
leaks = []
for c in feat_cols:
    a, b = base[c].iloc[:K+1].to_numpy(dtype=float), pert[c].iloc[:K+1].to_numpy(dtype=float)
    diff = ~np.isclose(a, b, equal_nan=True)
    if diff.any(): leaks.append((c, int(diff.sum())))
print("A) Registry compute_all: features at t<=K changed by future perturbation:", leaks or "NONE (all causal)")

# --- Test B: DataPipeline._process_raw_data (adds ffill().bfill()). Same perturbation.
dp = DataPipeline()
md1 = dp._process_raw_data(df.copy(), 'X', '1h')
md2 = dp._process_raw_data(df2.copy(), 'X', '1h')
F1 = pd.DataFrame(md1.features, columns=md1.feature_names); F2 = pd.DataFrame(md2.features, columns=md2.feature_names)
leaksB = {}
for c in md1.feature_names:
    diff = ~np.isclose(F1[c].iloc[:K+1], F2[c].iloc[:K+1], equal_nan=True)
    if diff.any(): leaksB[c] = int(diff.sum())
print(f"B) _process_raw_data (ffill+bfill): {len(leaksB)} features changed at t<=K; rows changed per feature (sample):", dict(list(leaksB.items())[:8]))

# --- Test C: where does bfill inject future info? Perturb only bar 250 and see which EARLIER rows change.
df3 = df.copy(); df3.loc[250, ['open','high','low','close']] *= 1.05
md3 = dp._process_raw_data(df3.copy(), 'X', '1h')
F3 = pd.DataFrame(md3.features, columns=md3.feature_names)
changed_rows = {}
for c in md1.feature_names:
    d = np.where(~np.isclose(F1[c].iloc[:250], F3[c].iloc[:250], equal_nan=True))[0]
    if len(d): changed_rows[c] = (int(d.min()), int(d.max()), len(d))
print("C) Perturb ONLY bar 250 -> earlier rows changed (feature: (first_row, last_row, n_rows)):")
for k,v in changed_rows.items(): print("   ", k, v)

# --- Warm-up NaN budget before bfill
nan_rows = base[feat_cols].isna().any(axis=1)
first_full = int(np.argmax(~nan_rows.to_numpy()))
print(f"Warm-up: rows with >=1 NaN feature before fill = {int(nan_rows.sum())}; first fully-valid row index = {first_full}")
worst = base[feat_cols].isna().sum().sort_values(ascending=False).head(6)
print("Features with most leading NaNs:", worst.to_dict())

# --- Non-stationary raw-level features (bounded vs unbounded)
lvl = [c for c in feat_cols if base[c].abs().max() > 100]
print("Features with |value|>100 (raw-level / cumulative, non-stationary):", lvl)

"""Reproduce the composite loss a NAIVE (unconditional) forecaster achieves on the
pipeline's own synthetic process, to compare against the recorded best_val_loss=0.0150."""
import numpy as np
rng = np.random.default_rng(0)
n = 50000; h = 12
rets = rng.normal(0.0001, 0.01, n)
prices = 1.1*np.exp(np.cumsum(rets))
y = (prices[h:] - prices[:-h]) / prices[:-h]          # exactly prepare_sequences target
def pinball(q, y, pred):
    e = y - pred
    return np.mean(np.maximum(q*e, (q-1)*e))
def composite(y, point, q10, q50, q90):
    mse = np.mean((y-point)**2)
    return mse + 0.5*(pinball(0.1,y,q10)+pinball(0.5,y,q50)+pinball(0.9,y,q90)), mse
# Oracle-unconditional baseline: predict sample mean / sample quantiles (constant)
c, mse = composite(y, y.mean(), *np.quantile(y,[0.1,0.5,0.9]))
z, msez = composite(y, 0.0, 0.0, 0.0, 0.0)
print(f"target std={y.std():.4f}  var={y.var():.2e}")
print(f"constant-quantile baseline composite loss = {c:.4f} (mse {mse:.2e})")
print(f"all-zeros baseline composite loss        = {z:.4f} (mse {msez:.2e})")
print("recorded best_val_loss in saved_models/training_history.json = 0.0150; checkpoints 0.0104-0.0160")

import logging, numpy as np, pandas as pd, warnings
logging.disable(logging.CRITICAL); warnings.filterwarnings("ignore")
from numpy.lib.stride_tricks import sliding_window_view
from evaluation.backtester import Backtester
from evaluation.metrics import MetricsCalculator

# 1) sliding_window_view shape as used in cli/system.py walk_forward_test.train_func
arr = np.zeros((1000, 37))  # 1000 bars, 5 OHLCV + 32 features
X = sliding_window_view(arr, window_shape=120, axis=0)
print("WF train_func X_train shape:", X.shape, "-> input_dim = X.shape[2] =", X.shape[2], "(should be 37)")

# 2) Backtester timestamps when df has RangeIndex (as built in cli/system.py:758)
n=600
rng=np.random.default_rng(0)
close=1.1*np.exp(np.cumsum(rng.normal(0,0.001,n)))
df=pd.DataFrame({'open':close,'high':close*1.001,'low':close*0.999,'close':close,'volume':1000.0})
i=[0]
def strat(data, **kw):
    i[0]+=1
    if i[0]%50==0: return {'action':'buy','stop_loss_pips':50,'take_profit_pips':100}
    if i[0]%50==25: return {'action':'close'}
    return {'action':'hold'}
bt=Backtester(initial_balance=10000, realistic_mode=False)
res=bt.run(df, strat, show_progress=False)
print("first 3 timestamps:", res.timestamps[:3])
print("inferred periods_per_year:", MetricsCalculator.infer_periods_per_year(res.timestamps))
print("sharpe:", res.sharpe_ratio, "ann_return:", res.annualized_return, "volatility:", res.volatility, "trades:", res.total_trades)
print("avg_trade_duration (hours):", res.avg_trade_duration)

# realistic mode: daily trade cap with wallclock timestamps
i[0]=0
bt2=Backtester(initial_balance=10000, realistic_mode=True)
res2=bt2.run(df, strat, show_progress=False)
print("realistic_mode trades over 600 bars (12 buy signals):", res2.total_trades)

# 3) Spread/slippage applied multiplicatively; check on USDJPY-like price with default pip_value
bt3=Backtester()
bt3.reset()
bt3._open_position('long', 150.0, pd.Timestamp('2024-01-01'), {'stop_loss_pips':50,'take_profit_pips':100})
p=bt3.positions[0]
print("USDJPY entry@150: entry_price=%.5f (spread+slip cost=%.5f price units = %.1f JPY pips), SL=%.4f (dist=%.1f JPY pips), size=%.0f units" % (p.entry_price, p.entry_price-150, (p.entry_price-150)/0.01, p.stop_loss, (150-p.stop_loss)/0.01, p.size))
bt4=Backtester(); bt4.reset()
bt4._open_position('long', 1.1, pd.Timestamp('2024-01-01'), {'stop_loss_pips':50,'take_profit_pips':100})
p=bt4.positions[0]
print("EURUSD entry@1.1: SL distance=%.1f pips (requested 50), size=%.0f units, notional=%.0f, risk at SL=%.2f (2%% of 10000=200)" % ((1.1-p.stop_loss)/0.0001, p.size, p.size*p.entry_price, (p.entry_price-p.stop_loss)*p.size))

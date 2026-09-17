import sys, logging; sys.path.insert(0, '/home/user/Leap')
logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd
from core.data_pipeline import DataPipeline
from evaluation.backtester import Backtester
from evaluation.metrics import MetricsCalculator
dp = DataPipeline(); md = dp.fetch_historical_data('EURUSD','1h', n_bars=400)
df = pd.DataFrame({'open': md.open,'high': md.high,'low': md.low,'close': md.close,'volume': md.volume})  # exactly as cli/system.py backtest() builds it
print("df index type passed to Backtester.run (cli/system.py:758):", type(df.index).__name__, "| has timestamp col:", 'timestamp' in df.columns)
def strat(data, **kw): return {'action': 'buy' if len(data)%50==0 else 'hold'}
bt = Backtester(); res = bt.run(df, strat, show_progress=False)
ts = pd.DatetimeIndex(res.timestamps); d = ts.to_series().diff().dropna()
print("backtest timestamps: span =", ts.max()-ts.min(), "| median delta =", d.median(), "| inferred periods/year =", MetricsCalculator.infer_periods_per_year(res.timestamps))
print("-> every bar stamped datetime.now(); annualized Sharpe/return use wall-clock deltas; daily-trade-limit resets never trigger.")
# Also check strategy input at bar i includes bar i itself (decision + fill at same close)
seen = {}
def strat2(data, **kw):
    seen['n'] = len(data); seen['last_close'] = data.iloc[-1]['close']; return {'action':'hold'}
bt.run(df.iloc[:10], strat2, show_progress=False)
print("at final loop iteration i=9: strategy sees len(data)=", seen['n'], "(includes bar i) and fills at bar i close ->", np.isclose(seen['last_close'], df.iloc[9]['close']))

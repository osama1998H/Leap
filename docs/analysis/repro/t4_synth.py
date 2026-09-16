import sys, logging, io, subprocess, os; sys.path.insert(0, '/home/user/Leap')
buf = io.StringIO(); h = logging.StreamHandler(buf); h.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.INFO, handlers=[h], format='%(levelname)s %(name)s: %(message)s')
import numpy as np, pandas as pd
from core import data_pipeline as dpm
from core.data_pipeline import DataPipeline
print("MT5_AVAILABLE on this host:", dpm.MT5_AVAILABLE)
dp = DataPipeline()
ok = dp.connect(); print("connect() returned:", ok, "| is_connected:", dp.is_connected)
md = dp.fetch_historical_data('EURUSD','1h', n_bars=2000)
print("fetch_historical_data returned MarketData? ", md is not None, "| bars:", len(md.close), "| features:", len(md.feature_names))
print("--- log lines emitted during connect+fetch (level: message) ---"); print(buf.getvalue().strip())
# synthetic stats
r = np.diff(np.log(md.close)); print("synthetic per-bar log-return std:", round(float(r.std()),5), "-> annualized (24*252 bars):", round(float(r.std()*np.sqrt(24*252)),3), "(real EURUSD 1h ~0.0007-0.001/bar, ~7-10%/yr)")
ts = pd.DatetimeIndex(md.timestamp); print("weekend bars present in synthetic:", int((ts.dayofweek>=5).sum()), "of", len(ts))
print("synthetic last timestamp == now (end_date=datetime.now()):", ts[-1])
# seed nondeterminism across processes (hash(timeframe))
code = "import sys;sys.path.insert(0,'/home/user/Leap');import logging;logging.disable(50);from core.data_pipeline import DataPipeline;from datetime import datetime;d=DataPipeline()._generate_synthetic_data(5, datetime(2024,1,1),'1h');print(round(float(d.close.iloc[-1]),8))"
outs = [subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,env={**os.environ,'PYTHONHASHSEED':str(s)}).stdout.strip() for s in ('0','1','2')]
print("synthetic close[-1] under PYTHONHASHSEED=0,1,2:", outs, "-> deterministic across processes:", len(set(outs))==1)
# global RNG side effect
np.random.seed(123); a = np.random.rand()
np.random.seed(123); DataPipeline()._generate_synthetic_data(10, pd.Timestamp('2024-01-01'), '1h'); b = np.random.rand()
print("np.random global state reseeded by _generate_synthetic_data (a!=b):", a!=b)

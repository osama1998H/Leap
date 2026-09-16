import logging, numpy as np, pandas as pd, warnings, sys
logging.disable(logging.CRITICAL); warnings.filterwarnings("ignore")
from numpy.lib.stride_tricks import sliding_window_view
from evaluation.backtester import Backtester
from evaluation.metrics import MetricsCalculator

print("=== 1) walk-forward train_func window shape ===")
arr = np.zeros((1000, 37))
X = sliding_window_view(arr, window_shape=120, axis=0)
print("X_train shape:", X.shape, "-> input_dim = X.shape[2] =", X.shape[2], "(true n_features=37)")

print("\n=== 2) Backtester on RangeIndex df (as cli/system.py builds it) ===")
n=600; rng=np.random.default_rng(0)
close=1.1*np.exp(np.cumsum(rng.normal(0,0.001,n)))
df=pd.DataFrame({'open':close,'high':close*1.001,'low':close*0.999,'close':close,'volume':1000.0})
i=[0]
def strat(data, **kw):
    i[0]+=1
    if i[0]%50==0: return {'action':'buy','stop_loss_pips':50,'take_profit_pips':100}
    if i[0]%50==25: return {'action':'close'}
    return {'action':'hold'}
bt=Backtester(initial_balance=10000); res=bt.run(df, strat, show_progress=False)
print("first 2 timestamps:", res.timestamps[:2])
print("inferred periods_per_year:", MetricsCalculator.infer_periods_per_year(res.timestamps))
print("sharpe=%.3g ann_return=%.3g volatility=%.3g trades=%d avg_dur_h=%.3g" % (res.sharpe_ratio,res.annualized_return,res.volatility,res.total_trades,res.avg_trade_duration))
i[0]=0; bt2=Backtester(initial_balance=10000, realistic_mode=True); res2=bt2.run(df, strat, show_progress=False)
print("realistic_mode trades over 600 bars (12 buy signals, max_daily_trades=5):", res2.total_trades)

print("\n=== 3) spread/SL math (multiplicative) ===")
bt3=Backtester(); bt3.reset()
bt3._open_position('long', 150.0, pd.Timestamp('2024-01-01'), {'stop_loss_pips':50,'take_profit_pips':100})
p=bt3.positions[0]
print("USDJPY@150 pip_value=0.0001: entry cost=%.1f JPY-pips, SL dist=%.1f JPY-pips (asked 50)" % ((p.entry_price-150)/0.01,(150-p.stop_loss)/0.01))
bt4=Backtester(); bt4.reset()
bt4._open_position('long', 1.1, pd.Timestamp('2024-01-01'), {'stop_loss_pips':50,'take_profit_pips':100})
p=bt4.positions[0]
print("EURUSD@1.1: SL dist=%.1f pips (asked 50), notional=%.0f on $10k (lev %.0fx)" % ((p.entry_price-p.stop_loss)/0.0001, p.size*p.entry_price, p.size*p.entry_price/10000))

print("\n=== 4) MetricsCalculator.calculate_all mutates periods_per_year ===")
mc=MetricsCalculator()
eq=np.linspace(10000,11000,100)
hourly=pd.date_range('2024-01-01',periods=100,freq='1h'); daily=pd.date_range('2024-01-01',periods=100,freq='1D')
a=mc.calculate_all(eq,timestamps=hourly)['annualized_return']; b=mc.calculate_all(eq,timestamps=daily)['annualized_return']
c=MetricsCalculator().calculate_all(eq,timestamps=daily)['annualized_return']
print("same instance hourly->daily ann_return: %.4g vs %.4g ; fresh instance daily: %.4g" % (a,b,c))

print("\n=== 5) Strategy: BrokerPosition passed as positions (live path) ===")
from core.strategy import CombinedPredictorAgentStrategy, StrategyConfig
from core.broker_interface import BrokerPosition
from datetime import datetime
bp=BrokerPosition(ticket=1,symbol='EURUSD',type=0,volume=0.1,price_open=1.1,price_current=1.1,sl=0,tp=0,profit=0,swap=0,commission=0,magic=1,comment='',time=datetime.now())
class FakeAgent:
    called=False
    def select_action(self, obs, deterministic=True):
        FakeAgent.called=True; return 3,0.0,0.0   # CLOSE
s=CombinedPredictorAgentStrategy(predictor=None, agent=FakeAgent(), config=StrategyConfig(lookback_window=5))
md=pd.DataFrame(np.random.rand(10,6),columns=['open','high','low','close','volume','f0'])
sig=s.generate_signal(md,[bp],symbol='EURUSD')
print("agent.select_action called:", FakeAgent.called, "| signal:", sig.action, "| agent_action:", sig.agent_action, "(agent wanted CLOSE)")
try:
    s._build_account_observation([bp]); print("no error")
except Exception as e: print("exception in _build_account_observation:", type(e).__name__, e)

print("\n=== 6) OrderManager volume with RiskManager (units treated as lots) ===")
from core.paper_broker import PaperBrokerGateway
from core.broker_interface import PaperBrokerConfig
from core.order_manager import OrderManager, TradingSignal, SignalType
from core.risk_manager import DynamicRiskManager
broker=PaperBrokerGateway(PaperBrokerConfig(initial_balance=10000)); broker.connect()
rm=DynamicRiskManager(initial_balance=10000)
om=OrderManager(broker, risk_manager=rm, default_risk_percent=0.01)
si=broker.get_symbol_info('EURUSD'); tick=broker.get_current_tick('EURUSD')
sig=TradingSignal(SignalType.BUY,'EURUSD',stop_loss_pips=50,take_profit_pips=100,risk_percent=0.01)
vol,sl,tp=om._calculate_position_params(sig,si,tick)
print("RiskManager.calculate_position_size ->", rm.calculate_position_size(tick.ask, sl), "(units); OrderManager volume ->", vol, "LOTS =", vol*100000, "units; notional $%.0f on $10k" % (vol*100000*tick.ask))
ex=om.execute_signal(sig); print("execute_signal ->", ex.executed, "|", ex.error_message)
om2=OrderManager(broker, risk_manager=None, default_risk_percent=0.01)
vol2,_,_=om2._calculate_position_params(sig,si,tick); print("without RiskManager volume ->", vol2, "lots (risk 1% of 10k over 50 pips = 0.02 lots expected)")
print("risk_manager notional on open (OrderManager:254):", vol*tick.ask*si.trade_contract_size, "| on close (OrderManager:330):", vol*tick.ask)
broker.disconnect()

print("\n=== 7) autotrade paper path -> which price provider? ===")
from core.broker_interface import create_broker
pc=PaperBrokerConfig(initial_balance=10000, leverage=100, magic_number=1, use_real_prices=True)
b=create_broker('paper', config=pc); print("price provider:", type(b._price_provider).__name__)

print("\n=== 8) OnlineLearningManager.step from AutoTrader (no log_prob/value) ===")
from training.online_learning import OnlineLearningManager
class A:
    experience_buffer=[]
    def store_transition(self,**k): A.experience_buffer.append(k)
olm=OnlineLearningManager(predictor=None, agent=A())
for _ in range(120):
    olm.step({'state':np.zeros(5),'features':np.zeros(5),'close':1.1,'returns':0.0}, prediction=0.001, actual=45.0, trading_action=1, trading_reward=50.0, done=True)
print("agent.experience_buffer after 120 closed trades:", len(A.experience_buffer), "| avg prediction_error:", np.mean(olm.prediction_errors))

print("\n=== 9) TradingEnvironment observation normalization: variance share ===")
from core.data_pipeline import DataPipeline
dp=DataPipeline(); md=dp.fetch_historical_data('EURUSD','1h',n_bars=1500)
ohlcv=np.column_stack([md.open,md.high,md.low,md.close,md.volume])
w=60; s0=500
pw=ohlcv[s0-w:s0].flatten(); fw=md.features[s0-w:s0].flatten(); obs=np.concatenate([pw,fw])
mu,sd=obs.mean(),obs.std()
print("n_features=%d; obs mean=%.1f std=%.1f" % (md.features.shape[1],mu,sd))
z=(obs-mu)/sd
prices_z=z[:w*5].reshape(w,5)[:,:4]
print("z-scored close range across 60-bar window: %.5f (std of price after norm)" % prices_z[:,3].std())
print("|z|>1 count among 300 OHLCV values: %d ; among %d feature values: %d" % ((np.abs(z[:w*5])>1).sum(), len(fw), (np.abs(z[w*5:])>1).sum()))
print("feature names with 200-window:", [f for f in md.feature_names if '200' in f])
print("NaN fraction in first 250 rows of features (before pipeline bfill this is where bfill fills from future):", np.isnan(dp.feature_engineer._registry.compute_all(pd.DataFrame({'timestamp':md.timestamp,'open':md.open,'high':md.high,'low':md.low,'close':md.close,'volume':md.volume}))[md.feature_names].values[:250]).mean().round(3))

print("\n=== 10) Transformer 'uncertainty' magnitude vs min_confidence=0.6 ===")
from models.transformer import TransformerPredictor
tp_=TransformerPredictor(input_dim=10, config={'d_model':16,'n_heads':2,'n_encoder_layers':1,'d_ff':32,'max_seq_length':20}, device='cpu')
Xr=np.random.randn(4,20,10).astype(np.float32)
r=tp_.predict(Xr, return_uncertainty=True); print("untrained uncertainty (q90-q10):", r['uncertainty'].ravel().round(4), "-> confidence=1-u ≈", (1-r['uncertainty'].ravel()).round(3))

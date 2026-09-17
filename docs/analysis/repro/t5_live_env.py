import sys, logging; sys.path.insert(0, '/home/user/Leap')
logging.disable(logging.CRITICAL)
import numpy as np
from core.paper_broker import PaperBrokerGateway
from core.data_pipeline import DataPipeline
from core.live_trading_env import LiveTradingEnvironment
from core.strategy import CombinedPredictorAgentStrategy, StrategyConfig
from core.trading_types import Action
broker = PaperBrokerGateway(); broker.connect()
dp = DataPipeline()  # not connected -> synthetic
W=60
env = LiveTradingEnvironment(broker=broker, symbol='EURUSD', data_pipeline=dp, window_size=W, feature_dim=None, match_training_obs=True)
obs, _ = env.reset()
print("after reset: price_buffer", len(env._price_buffer), "feature_buffer", len(env._feature_buffer), "| n_additional_features resolved:", env.n_additional_features, "| obs_dim:", env.observation_space.shape[0], "| actual features per bar:", len(env._feature_buffer[0]) if env._feature_buffer else None)
mdf = env.get_market_data(); print("get_market_data() columns (first 8):", list(mdf.columns[:8]), "... n_cols:", mdf.shape[1])
print("_fetch_initial_data hardcodes timeframe='1h'; last bar ts from pipeline:", dp.fetch_historical_data('EURUSD','1h',n_bars=5).timestamp[-1])
obs2, r, term, trunc, info = env.step(Action.HOLD)
print("after 1 step: price_buffer", len(env._price_buffer), "feature_buffer", len(env._feature_buffer), "-> lengths equal?", len(env._price_buffer)==len(env._feature_buffer))
mdf2 = env.get_market_data(); print("get_market_data() after step: n_cols =", mdf2.shape[1], "(features dropped when lengths differ)")
print("last appended 'bar' (bid,ask,bid,bid,0):", env._price_buffer[-1])
# Strategy path: feature_names from saved model (real names) vs live column names
class FakePred:
    input_dim = 5 + len(dp.feature_engineer.feature_names)
    def predict(self, X, return_uncertainty=False): return {'prediction': np.array([[0.01]]), 'uncertainty': 0.1}
class FakeAgent:
    def select_action(self, obs, deterministic=True):
        assert len(obs) == W*(5+len(dp.feature_engineer.feature_names))+8, f"obs dim {len(obs)}"
        return 1, 0.0, 0.0
strat = CombinedPredictorAgentStrategy(predictor=FakePred(), agent=FakeAgent(), config=StrategyConfig(lookback_window=W), feature_names=list(dp.feature_engineer.feature_names))
logging.disable(logging.NOTSET); logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
sig = strat.generate_signal(mdf, positions=[], symbol='EURUSD')
print("strategy w/ model feature_names on live df (feature_i cols) -> action:", sig.action, "pred_return:", sig.predicted_return, "agent:", sig.agent_action)
strat2 = CombinedPredictorAgentStrategy(predictor=FakePred(), agent=FakeAgent(), config=StrategyConfig(lookback_window=W), feature_names=None)
sig2 = strat2.generate_signal(mdf2, positions=[], symbol='EURUSD')
print("strategy auto-detect (AutoTrader default) on post-step df (no features) -> action:", sig2.action, "pred_return:", sig2.predicted_return, "agent:", sig2.agent_action)

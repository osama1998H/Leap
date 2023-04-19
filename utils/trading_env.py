import numpy as np
import tensorflow as tf
from tensorflow.python.framework import tensor_spec
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
import MetaTrader5 as mt5


class TradingEnv(py_environment.PyEnvironment):
    def __init__(self, symbol, lot_size, stop_loss, take_profit):
        super().__init__()

        self.symbol = symbol
        self.lot_size = lot_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.state = self.get_state()
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1, 6), dtype=np.float32, minimum=-np.inf, maximum=np.inf, name='observation')

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def get_state(self):
        positions = mt5.positions_get(symbol=self.symbol)
        if not positions:
            return np.zeros((1, 6))

        position = positions[0]
        volume = position.volume
        profit = position.profit
        spread = mt5.symbol_info_tick(
            self.symbol).ask - mt5.symbol_info_tick(self.symbol).bid

        if position.type == mt5.ORDER_TYPE_BUY:
            position_type = 0
        elif position.type == mt5.ORDER_TYPE_SELL:
            position_type = 1
        else:
            position_type = 2

        bid = mt5.symbol_info_tick(self.symbol).bid
        ask = mt5.symbol_info_tick(self.symbol).ask

        return np.array([[position_type, volume, profit, spread, bid, ask]], dtype=np.float32)

    def trade(self, action):
        if action == 0:
            mt5.positions_add(
                symbol=self.symbol,
                action=mt5.ORDER_TYPE_BUY,
                volume=self.lot_size,
                slippage=2,
                stoploss=self.stop_loss,
                takeprofit=self.take_profit
            )
        elif action == 1:
            mt5.positions_add(
                symbol=self.symbol,
                action=mt5.ORDER_TYPE_SELL,
                volume=self.lot_size,
                slippage=2,
                stoploss=self.stop_loss,
                takeprofit=self.take_profit
            )

    def get_reward(self, position_before, position_after):
        if not position_before and not position_after:
            return 0
        elif position_before and not position_after:
            return -1
        elif position_after and not position_before:
            return 1
        else:
            return position_after.profit - position_before.profit

    def is_done(self):
        positions = mt5.positions_get(symbol=self.symbol)
        if not positions:
            return True
        else:
            return False

    def _reset(self):
        self.state = self.get_state()
        return tf.reshape(self.state, self._observation_spec.shape)

    def _step(self, action):
        positions = mt5.positions_get(symbol=self.symbol)
        if not positions:
            return tensor_spec.termination(np.array([0, 0, 0, 0, 0, 0]), 0)

        position = positions[0]
        position_before = positions[0] if len(positions) > 0 else None
        self.trade(action)
        positions = mt5.positions_get(symbol=self.symbol)
        position_after = positions[0] if len(positions) > 0 else None
        next_state = tensor_spec.numpy_array_to_feature_dict(self.get_state())
        done = tensor_spec.termination(
            self.is_done(), reward=self.get_reward(position_before, position_after))
        reward = tensor_spec.scalar(
            self.get_reward(position_before, position_after))
        return tensor_spec.transition(next_state, reward=reward, discount=0.99, step_type=1 if done else 0)

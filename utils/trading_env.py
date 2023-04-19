import MetaTrader5 as mt5
import numpy as np


class TradingEnv:
    def __init__(self, symbol, lot_size, stop_loss, take_profit):
        self.symbol = symbol
        self.lot_size = lot_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_steps = 1000
        self.current_step = 0

    def get_state(self):
        positions = mt5.positions_get(symbol=self.symbol)
        if not positions:
            return np.zeros((1, 4))

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

        return np.array([[position_type, volume, profit, spread]])

    def get_reward(self, position_before, position_after):
        if position_after == None:
            reward = 0
        elif position_before != None and position_before.type == position_after.type:
            reward = -abs(position_after.profit - position_before.profit)
        else:
            reward = abs(position_after.profit)

        return reward

    def is_done(self):
        # Check if position is closed
        positions = mt5.positions_get(symbol=self.symbol)
        if len(positions) == 0:
            return True

        # Check if max steps have been reached
        if self.current_step >= self.max_steps:
            return True

        return False

    def step(self, action):
        positions = mt5.positions_get(symbol=self.symbol)
        if not positions:
            return np.array([0, 0, 0, 0, 0, 0])
        position = positions[0]
        position_before = positions[0] if len(positions) > 0 else None
        self.trade(action)
        positions = mt5.positions_get(symbol=self.symbol)
        position_after = positions[0] if len(positions) > 0 else None
        next_state = self.get_state()
        done = self.is_done()
        reward = self.get_reward(position_before, position_after)
        return next_state, reward, (done, None)


    def trade(self, action):
        if action == None:
            return

        position_before = mt5.positions_get(symbol=self.symbol)[0]
        if position_before != None and position_before.type == action:
            return

        request = {
            "action": action,
            "symbol": self.symbol,
            "volume": self.lot_size,
            "type": action,
            "deviation": 10,
            "magic": 123456,
            "comment": "AI Trading Bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
            "position": 0,
            "price": mt5.symbol_info_tick(self.symbol).bid if action == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(self.symbol).ask,
            "sl": mt5.symbol_info_tick(self.symbol).bid - self.stop_loss * mt5.symbol_info(self.symbol).point if action == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(self.symbol).ask + self.stop_loss * mt5.symbol_info(self.symbol).point,
            "tp": mt5.symbol_info_tick(self.symbol).bid + self.take_profit * mt5.symbol_info(self.symbol).point if action == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(self.symbol).ask - self.take_profit * mt5.symbol_info(self.symbol).point
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print("Failed to execute order. Error code:", result.retcode)

    def reset(self):
        mt5.shutdown()
        mt5.initialize()
        return self.get_state()

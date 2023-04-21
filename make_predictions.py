import numpy as np
from tensorflow.keras.models import load_model
from utils.trading_env import TradingEnv
from utils.trading_utils import normalize_data, prepare_data, denormalize_data
import MetaTrader5 as mt5

# set up the trading environment
symbol = "EURUSD"
lot_size = 0.01
stop_loss = 20
take_profit = 40
trading_env = TradingEnv(symbol, lot_size, stop_loss, take_profit)

# set up connection to MetaTrader 5
if not mt5.initialize():
    print("initialize() failed, error code =", mt5.last_error())
    quit()

# download historical data
rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 10000)
closing_prices = [rate[4] for rate in rates]

# normalize the data
scaler, data_normalized = normalize_data(closing_prices)

# prepare the data for training
window_size = 60
X, Y = prepare_data(data_normalized, window_size)

# reshape the data for LSTM input
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# load the trained model
model = load_model("path/to/saved/model")

# make predictions on the historical data
Y_pred = model.predict(X)

# denormalize the data
Y_pred = denormalize_data(scaler, Y_pred)

# disconnect from MetaTrader 5
mt5.shutdown()

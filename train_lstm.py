import numpy as np
from utils.trading_utils import create_model, normalize_data, prepare_data, train_model, predict, denormalize_data
import MetaTrader5 as mt5

# set up the trading environment
symbol = "EURUSD"

# set up connection to MetaTrader 5
if not mt5.initialize():
    print("initialize() failed, error code =", mt5.last_error())
    quit()

# download historical data
rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 50000)
print("rates", rates)
closing_prices = [rate[4] for rate in rates]
print("closing_prices", closing_prices)

# normalize the data
scaler, data_normalized = normalize_data(closing_prices)

# prepare the data for training
window_size = 60
X, Y = prepare_data(data_normalized, window_size)

# split the data into training and testing sets
split_ratio = 0.8
train_size = int(len(X) * split_ratio)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# reshape the data for LSTM input
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# create and train the LSTM model
model = create_model((X_train.shape[1], 1))
epochs = 10
train_model(model, X_train, Y_train, epochs)

# make predictions on the testing set
Y_pred = predict(model, X_test)

# denormalize the data
Y_test = denormalize_data(scaler, Y_test)
Y_pred = denormalize_data(scaler, Y_pred)

# calculate the accuracy of the model
accuracy = np.mean(np.abs((Y_test - Y_pred) / Y_test)) * 100
print("Accuracy: {:.2f}%".format(100 - accuracy))
print("prediction first", Y_pred[0])
print("prediction last", Y_pred[-1])
print("prediction all", Y_pred)

model.save(f"model-lstm.h5")

# disconnect from MetaTrader 5
mt5.shutdown()
print("shutdown")

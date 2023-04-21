from datetime import datetime
import numpy as np
from utils.trading_utils import create_model, normalize_data, prepare_data, train_model, predict, denormalize_data
import MetaTrader5 as mt5


# Set up connection to MetaTrader 5
if not mt5.initialize():
    print("initialize() failed, error code =", mt5.last_error())
    quit()

# symbols = mt5.symbols_get(group="*UR*")
symbols = mt5.symbols_get("*USD*")
symbols = ['EURUSD', 'GBPUSD', 'USDCHF', 'USDJPY', 'USDCNH', 'AUDUSD', 'NZDUSD', 'USDCAD']
# print('len(*RU*): ', len(symbols))
for s in symbols:
    print(s.name)
print("end")

# Download historical data for all symbols
closing_prices = []
for symbol in symbols:
    rates = mt5.copy_rates_from_pos(symbol.name, mt5.TIMEFRAME_D1, 0, 10000)
    closing_prices.append([rate[4] for rate in rates])

print(closing_prices)

# Normalize the data for all symbols
scalers = []
data_normalized = []
for prices in closing_prices:
    scaler, normalized = normalize_data(prices)
    scalers.append(scaler)
    data_normalized.append(normalized)

# Prepare the data for training for all symbols
window_size = 60
Xs, Ys = [], []
for data in data_normalized:
    X, Y = prepare_data(data, window_size)
    Xs.append(X)
    Ys.append(Y)

# Concatenate the data for all symbols
if closing_prices:
    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
else:
    print("No data available for the symbols.")


# Split the data into training and testing sets
split_ratio = 0.8
train_size = int(len(X) * split_ratio)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# Reshape the data for LSTM input
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Create and train the LSTM model
model = create_model((X_train.shape[1], 1))
epochs = 10
train_model(model, X_train, Y_train, epochs)

# Make predictions on the testing set
Y_pred = predict(model, X_test)

# Denormalize the data for all symbols
Y_test_denorm, Y_pred_denorm = [], []
for i in range(len(symbols)):
    Y_test_denorm.append(denormalize_data(
        scalers[i], Y_test[i * len(closing_prices):(i + 1) * len(closing_prices)]))
    Y_pred_denorm.append(denormalize_data(
        scalers[i], Y_pred[i * len(closing_prices):(i + 1) * len(closing_prices)]))

# Calculate the accuracy of the model for all symbols
for i in range(len(symbols)):
    accuracy = np.mean(
        np.abs((Y_test_denorm[i] - Y_pred_denorm[i]) / Y_test_denorm[i])) * 100
    print(
        "Symbol: {} - Accuracy: {:.2f}%".format(symbols[i].name, 100 - accuracy))

# Save the model

now = datetime.now()
model.save(f"model-mlsmt-{now}.h5")

# Disconnect from MetaTrader 5
mt5.shutdown()

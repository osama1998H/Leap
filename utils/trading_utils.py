import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def create_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=64, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(units=32, return_sequences=True))
    model.add(LSTM(units=16, return_sequences=False))
    model.add(Dense(units=1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def normalize_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(np.array(data).reshape(-1, 1))
    return scaler, data_normalized

def prepare_data(data, window_size):
    X, Y = [], []
    for i in range(len(data) - window_size - 1):
        X.append(data[i:(i + window_size), 0])
        Y.append(data[i + window_size, 0])
    return np.array(X), np.array(Y)

def train_model(model, X_train, Y_train, epochs):
    model.fit(X_train, Y_train, epochs=epochs, batch_size=32, verbose=1)

def predict(model, X):
    return model.predict(X)

def denormalize_data(scaler, data):
    return scaler.inverse_transform(data)


import matplotlib.pyplot as plt

def plot_learning_curve(scores, eps_history):
    fig, ax = plt.subplots(2)
    ax[0].plot(scores)
    ax[0].set_xlabel('Episode')
    ax[0].set_ylabel('Score')
    ax[1].plot(eps_history)
    ax[1].set_xlabel('Episode')
    ax[1].set_ylabel('Epsilon')
    fig.suptitle('Learning Curve')
    plt.show()


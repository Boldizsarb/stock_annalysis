import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from tensorflow.python.keras.layers import Dense

#from tensorflow.keras.models import Sequential
from keras.models import Sequential
#from tensorflow.keras.layers import LSTM, Dense, Dropout
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np


end_date = datetime.now()
start_date = end_date - timedelta(days=365)
interval = "1d"


def asml_prediction():

    ticker = 'ASML'
    ## everything all at once
    stock_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    selected_columns = stock_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    new_dataframe = pd.DataFrame(selected_columns)
    ############################ LSMT ######## LSMT ######## LSMT ######## LSMT ######## LSMT ######## LSMT
    # Ensure the index is a DatetimeIndex
    data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    data = data[['Close']]  # Use only the 'Close' prices

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Prepare training and testing datasets
    look_back = 60  # Number of previous days to consider for predicting the next day's price
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    # Compile and train the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=100, batch_size=32)

    # Predicting and inverse transformation to get actual values
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

    # Calculate RMSE to evaluate the model
    rmse = np.sqrt(mean_squared_error(y_test, predicted_stock_price))
    print(f"Root Mean Squared Error: {rmse}")

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(data[len(data) - len(y_test):].index, scaler.inverse_transform(y_test.reshape(-1, 1)), color='blue',
             label='Actual ASML Stock Price')
    plt.plot(data[len(data) - len(y_test):].index, predicted_stock_price, color='red',
             label='Predicted ASML Stock Price')
    plt.title('ASML Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('ASML Stock Price')
    plt.legend()
    plt.show()
#########################################################################################
def asml_prediction_forecast(days_to_forecast=30): ## the parameter is the what is the time interval, how long into the future!


    # Download and prepare data
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    data = data[['Close']]  # Use only the 'Close' prices

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Prepare training dataset
    look_back = 60
    X = []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i, 0])

    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Build the LSTM model
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])

    # Compile and train the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, scaled_data[look_back:], epochs=100, batch_size=32)

    # Forecasting future prices
    forecasted = []
    current_batch = scaled_data[-look_back:]

    for i in range(days_to_forecast):
        # Reshape current_batch to meet LSTM input requirements
        current_batch_reshaped = current_batch.reshape((1, look_back, 1))
        current_pred = model.predict(current_batch_reshaped)

        # Append the prediction and update current_batch
        forecasted.append(current_pred[0, 0])
        current_batch = np.append(current_batch[1:], current_pred[0, 0]).reshape(-1, 1)

    forecasted = scaler.inverse_transform(np.array(forecasted).reshape(-1, 1))

    # Preparing data for plotting
    last_date = data.index[-1]
    forecast_dates = pd.date_range(start=last_date, periods=days_to_forecast + 1)[1:]
    forecast_df = pd.DataFrame(forecasted, index=forecast_dates, columns=['Forecast'])

    # Plotting the results
    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'], label='Historical Prices')
    plt.plot(forecast_df['Forecast'], label='Forecasted Prices')
    plt.title('ASML Stock Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

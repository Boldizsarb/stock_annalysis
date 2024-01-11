import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
from tensorflow.python.keras.layers import Dense
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from statsmodels.tools.eval_measures import rmse
import warnings
import plotly.graph_objs as go
from sklearn.metrics import mean_squared_error, mean_absolute_error

### for tensorflow to work I had to roll back to python 11, and change bunch of packages to be competable!

end_date = datetime.now()
start_date = end_date - timedelta(days=365)
interval = "1d"
ticker = 'META'


def load_stock_data():
    stock_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    return stock_data


def lstm_prediction():  ### very first one



    ############################ LSMT ######## LSMT ######## LSMT ######## LSMT ######## LSMT ######## LSMT

    # data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    data = load_stock_data()
    data = data[['Close']]  # Use only the 'Close' prices

    # Data preprocessing
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    look_back = 60
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # splitting dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    # Compile and train
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=100, batch_size=32)

    # Prediction
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

    # Evaluation
    mse = mean_squared_error(y_test, predicted_stock_price)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predicted_stock_price)
    st.write(
        "The data in this script is trained on a neural network, specifically a type of recurrent neural network (RNN) known as Long Short-Term Memory (LSTM)")
    st.write(
        "LSTM is highly advantageous in stock price prediction as it retains short-term and long-term patterns, crucial for understanding erratic and non-linear fluctuations in stock prices.")

    # Plotting
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
    predicted_prices = predicted_stock_price
    dates = data[len(data) - len(y_test):].index

    # Creating a DataFrame for plotting
    plot_data = pd.DataFrame({
        'Actual Price': actual_prices.flatten(),
        'Predicted Price': predicted_prices.flatten()
    }, index=dates)

    # Plotting with Streamlit
    st.line_chart(plot_data)
    st.write(
        "This RMSE value is a crucial metric for evaluating the performance of the LSTM model. It gives a quantitative measure of how well the model is performing.")
    st.write(f"Mean Squared Error (MSE): {mse}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse}")
    st.write(f"Mean Absolute Error (MAE): {mae}")


#########################################################################################################################


def lstm_prediction_forecast(
        days_to_forecast):  ## the parameter is the what is the time interval, how long into the future!

    # data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    data = load_stock_data()
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
    st.write("Training model...")
    model.fit(X, scaled_data[look_back:], epochs=100, batch_size=32)

    # Forecasting future prices
    forecasted = []
    current_batch = scaled_data[-look_back:]

    for i in range(days_to_forecast):
        # current_batch needs to meet LSTM input requirements
        current_batch_reshaped = current_batch.reshape((1, look_back, 1))
        current_pred = model.predict(current_batch_reshaped)

        #  update current_batch
        forecasted.append(current_pred[0, 0])
        current_batch = np.append(current_batch[1:], current_pred[0, 0]).reshape(-1, 1)

    forecasted = scaler.inverse_transform(np.array(forecasted).reshape(-1, 1))

    #  plotting
    last_date = data.index[-1]
    forecast_dates = pd.date_range(start=last_date, periods=days_to_forecast + 1)[1:]
    forecast_df = pd.DataFrame(forecasted, index=forecast_dates, columns=['Forecast'])

    # Plotting  with Streamlit
    st.subheader(f'{ticker} Stock Price Forecast')
    chart_data = pd.concat([data['Close'], forecast_df['Forecast']], axis=1)
    st.line_chart(chart_data)
    st.write(
        "The LSTM forecasting algorithm is an advanced technique that utilizes the LSTM's capacity to comprehend and anticipate sequential data.")


def linear_regression():  # regression uses  'Close' price based on 'Open', 'High', 'Low', 'Volume'

    # stock_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    stock_data = load_stock_data()
    selected_columns = stock_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    new_dataframe = pd.DataFrame(selected_columns)
    ###########
    # Preparing the data for regression
    # We will predict 'Close' price based on 'Open', 'High', 'Low', 'Volume'

    X = new_dataframe[['Open', 'High', 'Low', 'Volume']]
    y = new_dataframe['Close']

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Create a linear regression model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Predicting the Stock prices
    y_pred = model.predict(X_test)

    # Combining the actual and predicted values for comparison
    comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    # Calculate MSE
    mse = mean_squared_error(y_test, y_pred)

    # Calculate RMSE
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    # Store these values in variables
    mse_value = mse
    rmse_value = rmse
    mae_value = mae

    df = pd.DataFrame(comparison_df,
                      index=pd.to_datetime(["2023-11-24", "2023-06-29", "2023-05-16", "2023-08-18", "2023-08-15"]))
    """
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Actual'], label='Actual', marker='o')
    plt.plot(df.index, df['Predicted'], label='Predicted', marker='x')
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()
    """
    st.write(
        "By including the 'Open,' 'High,' 'Low,' and 'Volume' data in addition to the 'Close' price, the model is equipped with a broader range of information regarding the stock's performance.")
    st.write(
        "The model has the ability to acquire knowledge from the associations among these characteristics and the target variable (in this instance, the 'Close' price) in order to enhance the precision of its predictions.")
    st.line_chart(comparison_df)
    st.write(f"Mean Squared Error (MSE): {mse_value}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse_value}")
    st.write(f"Mean Absolute Error (MAE): {mae_value}")


def linear_regression_forecast(days_to_forecast=30):
    stock_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    selected_columns = stock_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    new_dataframe = pd.DataFrame(selected_columns)
    print(new_dataframe)
    # days_to_forecast = 30
    # Normalize the data
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    data = data[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # preparation
    look_back = 60
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i, 0])  # Features (past values)
        y.append(scaled_data[i, 0])  # Target (current value)

    X, y = np.array(X), np.array(y)

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    #  Linear Regression model fitting
    model = LinearRegression()
    model.fit(X_train, y_train)

    forecasted = []
    current_batch = scaled_data[-look_back:]

    for i in range(days_to_forecast):
        current_pred = model.predict(current_batch.reshape(1, -1))[0]
        forecasted.append(current_pred)
        current_batch = np.roll(current_batch, -1)
        current_batch[-1] = current_pred

    # inverse transform to get original scale  ### the key work
    forecasted = scaler.inverse_transform(np.array(forecasted).reshape(-1, 1))
    """ ## with plt
    # Prepare data for plotting
    last_date = data.index[-1]
    forecast_dates = pd.date_range(start=last_date, periods=days_to_forecast + 1)[1:]
    forecast_df = pd.DataFrame(forecasted, index=forecast_dates, columns=['Forecast'])
    plot_data = pd.concat([data['Close'], forecast_df['Forecast']], axis=1)

    # Plotting with matplotlib
    plt.figure(figsize=(10, 5))
    plt.plot(plot_data['Close'], label='Actual Prices')
    plt.plot(plot_data['Forecast'], label='Forecasted Prices', linestyle='--')
    plt.title(f'{ticker} Stock Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    """
    # plot
    last_date = data.index[-1]
    forecast_dates = pd.date_range(start=last_date, periods=days_to_forecast + 1)[1:]
    forecast_df = pd.DataFrame(forecasted, index=forecast_dates, columns=['Forecast'])

    # vis with Streamlit
    st.subheader(f'{ticker} Stock Price Forecast')
    chart_data = pd.concat([data['Close'], forecast_df['Forecast']], axis=1)
    st.line_chart(chart_data)
    st.write(
        "Linear regression is commonly used in stock price forecasting when a simple and understandable model is required to estimate future prices. This technique involves fitting a linear equation to historical stock price data, assuming a linear relationship between past and future prices. By identifying a linear pattern in the data, the model can make predictions for future prices.")


def forest_model():  ##### FOREST FOREST ### FOREST FOREST ### FOREST FOREST ### FOREST FOREST ### FOREST FOREST ### FOREST FOREST

    stock_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    # selected_columns = stock_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    selected_columns = load_stock_data()
    new_dataframe = pd.DataFrame(selected_columns)
    print(new_dataframe)
    # Create a feature matrix (X) and target variable (y)
    X = new_dataframe.drop(columns=['Close'])
    y = new_dataframe['Close']

    # Split the data into training and testing sets
    train_size = int(0.8 * len(X))
    X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

    # Initialize and train the Random Forest Regressor
    random_forest_regressor = RandomForestRegressor(n_estimators=100, random_state=0)
    random_forest_regressor.fit(X_train, y_train)

    # predictions
    y_pred = random_forest_regressor.predict(X_test)
    ### ecalutaion metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    """
    # Visualize the results
    plt.figure(figsize=(16, 8))
    plt.title('Stock Price Prediction with Random Forest Regressor')
    plt.xlabel('Date')
    plt.ylabel('Close Price USD ($)')
    plt.plot(stock_data.index[train_size:], y_test, label='True Price')
    plt.plot(stock_data.index[train_size:], y_pred, label='Predicted Price')
    plt.legend()
    plt.show()
    """
    st.write(f'True vs. Predicted Stock Prices of {ticker}:')
    chart_data = pd.DataFrame({'True Price': y_test, 'Predicted Price': y_pred}, index=stock_data.index[train_size:])
    st.line_chart(chart_data)
    st.write(f"Mean Squared Error (MSE): {mse}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse}")
    st.write(f"Mean Absolute Error (MAE): {mae}")


def facebook2_model():
    stock_data = load_stock_data()
    stock_data.reset_index(inplace=True)
    data = stock_data[['Date', 'Close']]
    data = data.rename(columns={'Date': 'ds', 'Close': 'y'})

    # Create and fit the Prophet model
    prophet_model = Prophet()
    prophet_model.fit(data)

    # Create a dataframe for future dates
    future = prophet_model.make_future_dataframe(periods=365)  # Predict for one year into the future

    # Make predictions
    forecast = prophet_model.predict(future)

    # merging actual and predicted prices for the dates where actual prices are available
    merged_data = forecast.set_index('ds')[['yhat']].join(data.set_index('ds'))
    merged_data.dropna(inplace=True)  # Drop rows without actual prices
    # calculate evaluation metrics
    mse = mean_squared_error(merged_data['y'], merged_data['yhat'])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(merged_data['y'], merged_data['yhat'])

    """# Visualize the results with matplotlib
    plt.figure(figsize=(12, 6))
    plt.plot(data['ds'], data['y'], label='Actual Prices', color='blue')
    plt.plot(forecast['ds'], forecast['yhat'], label='Predicted Prices', color='red')
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='lightgray', alpha=0.5)
    plt.title('Stock Price Prediction with Prophet')
    plt.xlabel('Date')
    plt.ylabel('Close Price USD ($)')
    plt.legend()
    plt.grid(True)
    plt.show()
    """
    st.subheader(f"Facebook Prophet model forecast for {ticker}")
    st.write(
        "The Prophet model by Facebook recognizes uncertainties in time series data due to factors like seasonal patterns, holidays, and external events. To handle these uncertainties, Prophet includes uncertainty intervals")
    st.write(
        "A wider interval suggests higher uncertainty, while a narrower interval indicates greater confidence in the prediction")

    # Display
    st.write('Stock Price Prediction:')
    st.line_chart(forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']])
    st.write("**Interpretation:**")
    st.write(
        "**yhat:**  This is the central forecast or the predicted value in this case the stock price, for a specific date in the future.")
    st.write(
        "**yhat_lower:** The minimum expected value for a future variable is denoted by this. It provides a range within which the actual value is expected to fall with a certain level of certainty. It is calculated as a percentile of the projected values.")
    st.write(
        "**yhat_upper:** It represents the upper limit or expected maximum value for the desired variable on a future date. Similar to yhat_lower, it provides a range within which the actual value is expected to fall with a certain level of confidence")

    # Display the stock data
    st.write('Stock Data:')
    st.dataframe(stock_data)
    st.write("Evaluation Metrics:")
    st.write(f"Mean Squared Error (MSE): {mse}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse}")
    st.write(f"Mean Absolute Error (MAE): {mae}")


def arima(forecast_months=3):
    ## loading handling and cleaning data ->
    data = load_stock_data()
    # print(data)
    df = pd.DataFrame(data)
    selected_columns = df[['Close']]
    # print(selected_columns)
    selected_columns.index = pd.to_datetime(selected_columns.index)

    selected_columns = selected_columns.asfreq('D')  # 'D' for daily, adjust accordingly
    # Check for missing values
    if selected_columns['Close'].isna().any():
        print("Missing values found. Handling missing values...")

        selected_columns['Close'] = selected_columns['Close'].ffill()

        # selected_columns['Close'] = selected_columns['Close'].bfill()

    # decomposition
    # ETS Decomposition: The ETS Decomposition is used on time-series data to split error, trend and seasonality of the data.
    ets = seasonal_decompose(selected_columns['Close'], model='multiplicative')
    # ets.plot()
    # plt.show()

    ######################################

    # Fit auto_arima function
    warnings.filterwarnings("ignore")
    stepwise_fit = auto_arima(selected_columns['Close'], start_p=1, start_q=1,
                              max_p=3, max_q=3, m=12,
                              start_P=0, seasonal=True,
                              d=None, D=1, trace=True,
                              error_action='ignore',  # we don't want to know if an order does not work
                              suppress_warnings=True,  # we don't want convergence warnings
                              stepwise=True)

    stepwise_fit.summary()
    # print(stepwise_fit)

    # Extract ARIMA parameters
    arima_order = stepwise_fit.order
    seasonal_order = stepwise_fit.seasonal_order

    arima_params = (arima_order, seasonal_order)

    print("ARIMA parameters:", arima_params)

    print("done1")

    # Split data into train / test sets
    train = selected_columns.iloc[:len(selected_columns) - 12]
    test = selected_columns.iloc[len(selected_columns) - 12:]  # set one year(12 months) for testing

    # fitting the extracted best model values to the model
    model = SARIMAX(selected_columns['Close'],
                    order=arima_order,
                    seasonal_order=seasonal_order)

    result = model.fit()
    result.summary()
    start = len(train)
    end = len(train) + len(test) - 1
    # Predictions for one-year against the test set and the rmse being measured on this one!

    predictions = result.predict(start, end,
                                 typ='levels').rename("Predictions")

    # plot predictions and actual values
    predictions  # .plot(legend=True)
    test['Close']  # .plot(legend=True)
    # plt.show()
    # plt.show()

    # Calculate root mean squared error
    print("rmse = ", rmse(test["Close"], predictions))

    # Calculate mean squared error
    print("mse = ", mean_squared_error(test["Close"], predictions))

    # Train the model on the full dataset
    model = SARIMAX(selected_columns['Close'],
                    order=(1, 0, 0),
                    seasonal_order=(2, 1, 0, 12))

    # Forecast for the next 3 years
    forecast = result.predict(start=len(selected_columns),
                              end=(len(selected_columns) - 1) + forecast_months * 12,  ## increment it by month
                              typ='levels').rename('Forecast')

    # evaluation metrics:
    rmse_value = rmse(test["Close"], predictions)
    mse_value = mean_squared_error(test["Close"], predictions)
    mae_value = mean_absolute_error(test["Close"], predictions)
    """
    # Plot the forecast values
    selected_columns['Close'].plot(figsize=(12, 5), legend=True)
    forecast.plot(legend=True)
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 5))
    selected_columns['Close'].plot(ax=ax, legend=True)
    forecast.plot(ax=ax, legend=True)
    ax.set_title("Close Price and Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")

    # Use Streamlit to render the plot
    st.pyplot(fig)
    """
    # Plot using Plotly
    st.write("ARIMA stands for AutoRegressive Integrated Moving Average often used for time series forecasting")
    st.write(
        "he ARIMA model has three main parameters includes p lag observations, d differences, and a moving average window of size q.")
    st.write(
        "These parameters are being determined by testing the models first, and choosing the most suitable for the data.")
    fig = go.Figure()
    # Historical data
    fig.add_trace(
        go.Scatter(x=selected_columns.index, y=selected_columns['Close'], mode='lines', name='Historical Close Price'))
    # Forecast data
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast, mode='lines', name='Forecast', line=dict(dash='dash')))

    fig.update_layout(title=f'{ticker}Stock Price Forecast', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig, use_container_width=True)  ## vis

    st.write(f"Root Mean Squared Error (RMSE): {rmse_value}")
    st.write(f"Mean Squared Error (MSE): {mse_value}")
    st.write(f"Mean Absolute Error (MAE): {mae_value}")
    st.write(
        "RMSE and MSE are measures of the average error between the predicted values and the actual values. They are widely used to evaluate the accuracy of forecasting models. Lower values of RMSE and MSE indicate better model performance")
    st.write(
        "The Values are being measured on the trained data rather than the forecasted one to not influence the measurements.")

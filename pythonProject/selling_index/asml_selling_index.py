import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from textblob import TextBlob

from correlation import calculate_asml_correlation

from tensorflow.python.keras.layers import Dense
#from tensorflow.keras.models import Sequential
from keras.models import Sequential
#from tensorflow.keras.layers import LSTM, Dense, Dropout ## doesnt work
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import plotly.graph_objs as go
import requests


end_date = datetime.now()
start_date = end_date - timedelta(days=365)
interval = "1d"
ticker = 'ASML'

def load_stock_data():
    stock_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    return stock_data

def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment



def asml_prediction_forecast2(days_to_forecast): ## the parameter is the what is the time interval, how long into the future!

    API_KEY = 'KAM4TO0GVBQT9RSA'
    URL = "https://www.alphavantage.co/query"

    def news(symbol2): #### news api

        API_KEY = "a55e74c97ad14f7aa13dca0a4a0d0f04"

        # NewsAPI URL
        URL = "https://newsapi.org/v2/everything"
        company_name = f"{symbol2}"
        params = {
            "q": company_name,
            "apiKey": API_KEY,
            "language": "en",
            "sortBy": "publishedAt"
        }
        response = requests.get(URL, params=params)
        news_data = response.json()

        # Display news articles
        if news_data.get("articles"):
            for article in news_data["articles"]:
                with st.container():
                    # Display the article's title as a clickable link
                    st.markdown(f"### [{article.get('title', 'No Title')}]({article.get('url', '#')})",
                                unsafe_allow_html=True)

                    article_title = article.get('title', 'No Title')
                    sentiment = analyze_sentiment(article_title)

                    # Display the author
                    st.write(f"**Author:** {article.get('author', 'Unknown Author')}")

                    # Display the published date
                    st.write(f"**Published At:** {article.get('publishedAt', 'Unknown Date')}")
                    st.write(f"Sentiment - Polarity: {sentiment.polarity}, Subjectivity: {sentiment.subjectivity}")
                    if sentiment.polarity > 0.04:
                        sentiment_label = "The news likely to have a Positive impact on the stock price. "
                    elif sentiment.polarity < -0.04:
                        sentiment_label = "The news likely to have a Negative impact on the stock price"
                    else:
                        sentiment_label = "The news likely to have a Neutral impact on the stock price"
                    st.write(f"{sentiment_label}")
                    st.write("---")
        else:
            st.write("No news articles found for the company.")

    def info(symbol):
        st.title(f"Company Data - {symbol}")

        # Fetch company data
        stock = yf.Ticker(symbol)
        company_data = stock.info

        # Display selected company data
        st.header(f"{symbol} Overview")
        st.write("Name:", company_data.get("longName", "N/A"))
        st.write("Exchange:", company_data.get("exchange", "N/A"))
        st.write("Industry:", company_data.get("industry", "N/A"))
        st.write("Description:", company_data.get("longBusinessSummary", "N/A"))
        st.write("Market Capitalization:", company_data.get("marketCap", "N/A"))
        st.write("PE Ratio:", company_data.get("trailingPE", "N/A"))
        st.write("Dividend Yield:",
                 company_data.get("dividendYield", "N/A") * 100 if company_data.get("dividendYield") else "N/A")
        st.write("52 Week High:", company_data.get("fiftyTwoWeekHigh", "N/A"))
        st.write("52 Week Low:", company_data.get("fiftyTwoWeekLow", "N/A"))

        # Current stock price is part of the 'info' dictionary
        st.header(f"Current Stock Performance of {symbol}")
        st.write("Current Price:", company_data.get("currentPrice", "N/A"))
        st.write("Previous Close:", company_data.get("previousClose", "N/A"))
        st.write("Change:", company_data.get("regularMarketChange", "N/A"))
        st.write("Change Percent:", company_data.get("regularMarketChangePercent", "N/A"))
        st.write("It is also important to learn about the company, here is a little help with it:")
        st.write("---")

        ###### news api call
        news(symbol)
        ############################## info over

    #data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
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
    st.write("The program retrieves the last stock price. Computes the 15 day moving average.")
    st.write("The program also calculates Relative Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements.")
    st.write("Consider the threshold and trend when making a decision. If the predicted price is much higher than the actual price and the recent trend is positive, buy. If the predicted price is much lower than the actual price and the recent trend is negative, sell. If the predicted price change falls within the threshold or the trend contradicts the prediction, hold.")
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
    #############################################################
    latest_stock_price = data['Close'].iloc[-1]
    #print(f"Latest stock price: {latest_stock_price}")
    formatted_latest_stock_price = "{:.2f}".format(latest_stock_price)
    # Get the last forecasted amount and store it in a variable
    last_forecasted_amount = forecasted[-1, 0]
    formatted_last_forecasted_amount = "{:.2f}".format(last_forecasted_amount)

    #print(f"Last forecasted amount: {last_forecasted_amount}")
    # threshold for decision making (e.g., 5%)
    threshold = 0.03
    # Trend Analysis
    # Calculate the moving average for a certain period (e.g., 30 days)
    moving_average_period = 30  ## a month
    moving_average = data['Close'].rolling(window=moving_average_period).mean().iloc[-1]
    short_term_moving_average = data['Close'].rolling(window=15).mean().iloc[-1]
    ######## RSI   ### RSI   ### RSI   ### RSI   ### RSI   ### RSI   ### RSI   ### RSI
    # Adding a technical indicator like RSI
    def calculate_rsi(data, window=14):
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    rsi = calculate_rsi(data).iloc[-1]

    # print(decision)
    text_color = "red"
    st.markdown(f'<p>The  stock price is: <span style="color:{text_color};">{formatted_latest_stock_price}</span></p>', unsafe_allow_html=True)
    st.markdown(f'<p>After {days_to_forecast} days, the predicted price will be: <span style="color:{text_color};">{formatted_last_forecasted_amount}</span></p>',
                unsafe_allow_html=True)
    #st.write(f"The **latest** stock price is: {formatted_latest_stock_price}")
    #st.write(f"After {days_to_forecast} days, the **predicted** price will be: {formatted_last_forecasted_amount}")

    # Decision Making
    if last_forecasted_amount > latest_stock_price * (1 + threshold):
        if latest_stock_price > short_term_moving_average and rsi < 70:  # Avoiding overbought situations
            decision = "Buy - Positive short-term trend and not overbought"
        else:
            decision = "Hold - Conditions for buying not met"
    elif last_forecasted_amount < latest_stock_price * (1 - threshold):
        if latest_stock_price < short_term_moving_average and rsi > 30:  # Avoiding oversold situations
            decision = "Sell - Negative short-term trend and not oversold"
        else:
            decision = "Hold - Conditions for selling not met"
    else:
        decision = "Hold - Predicted price within threshold limits"



    st.write("After the trend analysis with a calculated 30 days avarage, considering the 3% percent threshold, you should:")
    st.subheader(decision)
    # Plotting the predicted stock price using Plotly
    last_date = data.index[-1]
    prediction_dates = pd.date_range(start=last_date, periods=days_to_forecast + 1)
    predicted_prices = forecasted.flatten()

    # Create a Plotly figure
    fig = go.Figure()

    # Add traces for actual and predicted prices
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Actual Prices'))
    fig.add_trace(go.Scatter(x=prediction_dates, y=predicted_prices, mode='lines+markers', name='Predicted Prices'))

    # Update layout
    fig.update_layout(
        title=f"{ticker} Stock Price Prediction for the Next {days_to_forecast} Days",
        xaxis_title='Date',
        yaxis_title='Price',
        legend_title='Legend'
    )
    ############## selll ### buyy#######

    top_correlated, least_correlated = calculate_asml_correlation()
    # Retrieve the top 3 correlated companies
    top_3_correlated_companies = top_correlated.head(3)
    company_names = list(top_3_correlated_companies.keys())
    st.write(company_names[1])

    def sell():
        st.markdown(f'<h2><span style="color:{text_color};">Sell!</span></h2>', unsafe_allow_html=True)

        st.write("You may wish to consider examining the subsequent companies as a result of the strong correlation; it is likely that the stock prices of these companies will also experience a negative impact.")
        st.write("The Information was provided by ALPHA VANTAGE *(API)*")
        st.write("Highly correlated companies:")
        with st.expander(f"{company_names[0]}"):
            info(company_names[0])
        with st.expander(f"{company_names[1]}"):
            info(company_names[1])
        with st.expander(f"{company_names[2]}"):
            info(company_names[2])

    def buy():
        st.markdown(f'<h2><span style="color:green;">Buy!</span></h2>', unsafe_allow_html=True)
        st.write(
            "You may wish to consider examining the subsequent companies as a result of the strong correlation; it is likely that the stock prices of these companies will also experience a positive impact.")
        st.write("The Information was provided by ALPHA VANTAGE *(API)*")
        st.write("Highly correlated companies:")
        with st.expander(f"{company_names[0]}"):
            info(company_names[0])
        with st.expander(f"{company_names[1]}"):
            info(company_names[1])
        with st.expander(f"{company_names[2]}"):
            info(company_names[2])


    # ###############################Display
    st.plotly_chart(fig)
    if decision == ("Hold - Predicted price within threshold limits" or "Hold - Conditions for selling not met" or "Hold - Conditions for buying not met"):
        st.write("Alternatively, if you do not want to hold: ")
        if formatted_latest_stock_price > formatted_last_forecasted_amount:
            st.write("You might consider selling since the forecasted stock price is lower.")
            sell()
        elif formatted_latest_stock_price < formatted_last_forecasted_amount:
            st.write("You might consider buying since the forecasted stock price is higher.")
            buy()
    elif decision == "Buy - Positive short-term trend and not overbought":
        #st.write(" Buy")
        buy()
    elif decision == "Sell - Negative short-term trend and not oversold":
        #st.write("sell ")
        sell()

















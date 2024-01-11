
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px


########################## NEED TO REPLICATE THE FILE FOR EACH  COMPANY #############################
end_date = datetime.now()
start_date = end_date - timedelta(days=365)
interval = "1d"

def NVDA_EDA():

    ticker = 'NVDA'
    ## individual variables
    close_data = yf.download(ticker, start=start_date, end=end_date, interval="1d")['Close']
    opening_data = yf.download(ticker, start=start_date, end=end_date, interval="1d")['Open']
    print(close_data)
    print(opening_data)



    ## everything all at once
    stock_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    selected_columns = stock_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    new_dataframe = pd.DataFrame(selected_columns)

    ########################## Temporal Structure Plot   ############## Temporal Structure Plot   ############## Temporal Structure Plot   ############## Temporal Structure Plot

    with st.expander("Temporal Structure"):

        st.subheader(f'Temporal Structure of {ticker} Stock Data')
        st.write("This refers to the organization, patterns, and trends that data exhibits over time. In the context of time series data, which is data collected "
                 "or recorded at regular intervals over a sequence of time points, the temporal structure refers to how the values in the dataset change and evolve as time progresses.")
        plt.figure(figsize=(12, 8))

        ### plot with plt
        """
        for column in selected_columns.columns:
            plt.plot(selected_columns.index, selected_columns[column], label=column)

        plt.title(f'Temporal Structure of {ticker} Stock Data')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        st.pyplot(plt)
        """
        st.line_chart(selected_columns) ## with streamlite

    ############################### Moving Averages ########################## Moving Averages ########################## Moving Averages
    #st.write("---")
    with st.expander("Moving Averages"):
        st.subheader('Moving Averages')
        st.write("Moving averages smooth out price data, reducing noise and making it easier to discern trends and patterns."
                 " Traders often use moving averages as part of their trading strategies.")
        st.write("**SMA 50**  represents the average closing price of the stock over the last 50 trading days.")
        st.write("**SMA 200** represents the average closing price of the stock over the last 200 trading days.")
        for window in [50, 200]:
            selected_columns['SMA_' + str(window)] = selected_columns['Close'].rolling(window=window).mean()

        """ ## plotting with plt
        plt.figure(figsize=(12, 8))
        for window in [50, 200]:
            plt.plot(selected_columns.index, selected_columns['SMA_' + str(window)], label=f'SMA {window} Days')

        plt.title(f'Moving Averages for {ticker} Stock Data')
        plt.xlabel('Date')
        plt.ylabel('Moving Average')
        plt.legend()
        st.pyplot(plt)
        #st.write("---")
        """
        ## with steamlite
        moving_avg_data = selected_columns[['SMA_50', 'SMA_200']]
        st.line_chart(moving_avg_data)

    ################################## Changes in distribution
    with st.expander("Changes in Distributions"):
        st.subheader('Changes in Distributions')
        st.write("The x-axis represents the values or ranges of the data, and the y-axis represents the frequency or count.")
        st.write("Histograms provide insight into the data's central tendency, spread, and the presence of outliers.")
        st.write("This aproach analyzes the data as a continuous time series without breaking it down into specific intervals.")

        st.write("Histograms and density plots for selected columns are displayed below.")

        """ ## with plt
        for i, column in enumerate(selected_columns.columns):
            plt.figure(figsize=(8, 4))
            sns.histplot(selected_columns[column], kde=True)
            plt.title(f'Distribution of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.tight_layout()
            st.pyplot(plt)
        """
        # with plotly
        for column in selected_columns.columns:
            # Create and display histogram using Plotly Express
            fig = px.histogram(selected_columns, x=column, nbins=30)
            st.plotly_chart(fig, use_container_width=True)
        st.write("This can help identify patterns or anomalies in the data, such as unusual trading volumes or price movements.")
       # st.write("---")
    ################################### changes in distributions in intervals #########################################
    with st.expander("Changes in Distributions Over monthly Intervals"):
        #resampled_data = stock_data.resample('M').last()  # changing the for quarterly intervals to months
        st.subheader(f'Changes in Distributions Over monthly Intervals for {ticker} Stock Data')
        st.write("This approach provides a temporal understanding of how the distribution of stock data evolves over time intervals.")
        st.write(" This more granular view showcases the distribution of stock data changes within each interval, making it easier to identify short-term trends and seasonal patterns.")
        """ ### with plt
        for column in stock_data.columns:
            #  list for each interval
            data_to_plot = [interval_data[column] for _, interval_data in resampled_data.iterrows()]

            # math for the statistic
            means = [interval_data.mean() for interval_data in data_to_plot]
            std_devs = [interval_data.std() for interval_data in data_to_plot]
            intervals = [interval_start.strftime('%b %Y') for interval_start, _ in resampled_data.iterrows()]

            #### visualizing
            plt.figure(figsize=(8, 4))
            plt.errorbar(intervals, means, yerr=std_devs, marker='o', label=column)

            plt.title(f'Distribution of {column}')
            plt.xlabel('Time Interval')
            plt.ylabel(column)
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            st.pyplot(plt)
        """
        # Resample the data monthly and calculate mean and standard deviation
        monthly_resampled = stock_data.resample('M').agg(['mean', 'std'])

        for column in stock_data.columns:
            # Ensure column exists in the resampled data
            if column in monthly_resampled:
                # Extracting the mean and std for the column
                mean_series = monthly_resampled[column]['mean']
                std_series = monthly_resampled[column]['std']

                # Create a DataFrame for plotting
                interval_df = pd.DataFrame({
                    'Date': mean_series.index,
                    'Mean': mean_series,
                    'Standard Deviation': std_series
                })

                # Create and display the error bar plot using Plotly Express
                fig = px.scatter(interval_df, x='Date', y='Mean', error_y='Standard Deviation',
                                 labels={'Mean': f'Mean {column}'}, title=f'Distribution of {column} Over Time')
                fig.update_layout(xaxis_title='Time Interval', yaxis_title='Value', showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    #st.write("---")
    ############################ Volatility analysis    ######### Volatility analysis    ######### Volatility analysis    ######### Volatility analysis
    with st.expander("Volatility Analysis"):
        # calculate daily returns from the stock's closing prices
        rolling_window = 30
        stock_data['Daily_Return'] = stock_data['Close'].pct_change().fillna(0) ## based on closed price

        #  rolling standard deviation of returns --> very important
        stock_data['Rolling_Volatility'] = stock_data['Daily_Return'].rolling(window=rolling_window).std()


        st.subheader('Volatility Analysis')
        st.write(f"The line plot shows how the volatility of {ticker} stock returns changes over time. In this case, "
                 f"it's computed over 30-day intervals. A rising trend in the rolling volatility indicates"
                 f" increasing price fluctuations or risk, while a falling trend suggests decreasing risk.")
        st.write(f"")
        """ ## with plt
        plt.figure(figsize=(12, 6))
        plt.plot(stock_data.index, stock_data['Rolling_Volatility'], label=f'{rolling_window}-Day Rolling Volatility',
                 color='b')
        plt.title(f'Rolling Volatility of {ticker} Stock Returns')
        plt.xlabel('Date')
        plt.ylabel('Volatility (Standard Deviation)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(plt)
        #st.write("---")
        """
        ## with streamlit:
        st.line_chart(stock_data['Rolling_Volatility'])
    ############################### RSI Analysis
    with st.expander("Relative Strength Index (RSI) Analysis"):

        window = 14

        delta = stock_data['Close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        stock_data['RSI'] = rsi

        st.subheader('Relative Strength Index (RSI) Analysis')
        st.write(f"Visualizing the Relative Strength Index (RSI) of {ticker} over the past year.")
        st.write("RSI is a widely used technical indicator in stock market analysis. It ranges from 0 to 100 and helps traders and investors identify potential overbought and oversold conditions in a stock")
        st.write("Above **70** are typically considered overbought, suggesting that the stock may be due for a pullback")
        st.write("Below **30** are typically considered oversold, indicating that the stock may be due for a rebound.")
        st.write("RSI can also help confirm the strength of a trend. Rising RSI values during an uptrend indicate strong buying pressure")
        # visualizing
        rsi_chart_data = pd.DataFrame({
            'RSI': stock_data['RSI'],
            'Overbought (70)': 70,
            'Oversold (30)': 30
        }, index=stock_data.index)

        # Visualizing with Streamlit
        st.line_chart(rsi_chart_data)
        st.write("Traders often use RSI to generate buy and sell signals.")
        #st.write("---")


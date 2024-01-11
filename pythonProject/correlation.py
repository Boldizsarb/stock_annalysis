import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
import seaborn as sns

## the symbols of all the companies
nasdaq_100_tickers = [
        'AAPL', 'MSFT', 'AMZN', 'AVGO', 'META', 'TSLA', 'NVDA', 'GOOGL', 'GOOG', 'COST',
        'ADBE', 'AMD', 'PEP', 'NFLX', 'INTC', 'CSCO', 'TMUS', 'CMCSA', 'INTU', 'QCOM',
        'TXN', 'AMGN', 'HON', 'AMAT', 'BKNG', 'ISRG', 'SBUX', 'VRTX', 'LRCX', 'GILD',
        'ADI', 'MDLZ', 'PDD', 'MU', 'ADP', 'PANW', 'REGN', 'MELI', 'KLAC', 'SNPS',
        'CDNS', 'CSX', 'PYPL', 'ASML', 'MAR', 'LULU', 'CTAS', 'NXPI', 'MNST', 'ABNB',
        'CRWD', 'ROP', 'CHTR', 'WDAY', 'ORLY', 'MRVL', 'ADSK', 'PCAR', 'MCHP', 'DXCM',
        'CPRT', 'ROST', 'KDP', 'IDXX', 'FTNT', 'ODFL', 'KHC', 'PAYX', 'AEP', 'AZN',
        'MRNA', 'BIIB', 'CTSH', 'TEAM', 'CEG', 'DDOG', 'FAST', 'DASH', 'EA', 'ON',
        'CSGP', 'GEHC', 'EXC', 'BKR', 'VRSK', 'GFS', 'XEL', 'ZS', 'TTD', 'ANSS',
        'DLTR', 'CDW', 'CCEP', 'MDB', 'FANG', 'WBD', 'TTWO', 'SPLK', 'WBA', 'ILMN', 'SIRI'
    ]


## retrieval dates as global variables
end_date = datetime.now()
start_date = end_date - timedelta(days=365)


def calculate_asml_correlation():
    asml_ticker = 'ASML'
    nasdaq_100_tickers.append(asml_ticker)

    # Download historical stock data for all NASDAQ 100 tickers
    nasdaq_100_data = yf.download(nasdaq_100_tickers, start=start_date, end=end_date,interval="1d")['Close']

    # Calculate the correlation matrix
    correlation_matrix = nasdaq_100_data.corr()

    # Get the correlation of ASML with other companies
    asml_correlations = correlation_matrix[asml_ticker].drop(asml_ticker)

    # Sort the correlations
    sorted_asml_correlations = asml_correlations.sort_values(ascending=False)

    return sorted_asml_correlations.head(10), sorted_asml_correlations.tail(10)

def calculate_meta_correlaiton():
    meta_ticker = 'META'  # Change to META

    # Include META in the list of tickers
    nasdaq_100_tickers.append(meta_ticker)

    # Download historical stock data for all NASDAQ 100 tickers
    nasdaq_100_data = yf.download(nasdaq_100_tickers, start=start_date, end=end_date,interval="1d")['Close']

    # Calculate the correlation matrix
    correlation_matrix = nasdaq_100_data.corr()

    # Get the correlation of META with other companies
    meta_correlations = correlation_matrix[meta_ticker].drop(meta_ticker)

    # Sort the correlations
    sorted_meta_correlations = meta_correlations.sort_values(ascending=False)

    return sorted_meta_correlations.head(10), sorted_meta_correlations.tail(10)


def calculate_bkng_correlation():
    bkng_ticker = 'BKNG'

    # Include BKNG in the list of tickers
    nasdaq_100_tickers.append(bkng_ticker)

    # Download historical stock data for all NASDAQ 100 tickers
    nasdaq_100_data = yf.download(nasdaq_100_tickers, start=start_date, end=end_date,interval="1d")['Close']

    # Calculate the correlation matrix
    correlation_matrix = nasdaq_100_data.corr()

    # Get the correlation of BKNG with other companies
    bkng_correlations = correlation_matrix[bkng_ticker].drop(bkng_ticker)

    # Sort the correlations
    sorted_bkng_correlations = bkng_correlations.sort_values(ascending=False)

    return sorted_bkng_correlations.head(10), sorted_bkng_correlations.tail(10)


def calculate_nvda_correlation():
    nvda_ticker = 'NVDA'

    nasdaq_100_tickers.append(nvda_ticker)
    #  historical stock data for all NASDAQ 100 tickers
    nasdaq_100_data = yf.download(nasdaq_100_tickers, start=start_date, end=end_date,interval="1d")['Close']

    # the correlation matrix
    correlation_matrix = nasdaq_100_data.corr()

    # Get the correlation of NVDA with other companies
    nvda_correlations = correlation_matrix[nvda_ticker].drop(nvda_ticker)

    # Sort the correlations
    sorted_nvda_correlations = nvda_correlations.sort_values(ascending=False)

    return sorted_nvda_correlations.head(10), sorted_nvda_correlations.tail(10)

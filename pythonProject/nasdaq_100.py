import yfinance as yf
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
from datetime import datetime, timedelta


###### the 100 top companies
### task 2     ### task 2     ### task 2     ### task 2     ### task 2
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

## Download 1-year data for each company

end_date = datetime.now()  ##  not used here
start_date = end_date - timedelta(days=365)

def download_stock_data( start, end):
    stock_data = {}
    for ticker in nasdaq_100_tickers:
        try:
            data = yf.download(ticker, start=start, end=end, interval="1d")
            if not data.empty:
                stock_data[ticker] = data['Close'] ### close is the feature it was clustered upon
        except Exception as e:
            print(f"Error downloading data for {ticker}: {e}")
    return pd.DataFrame(stock_data)

# Function to preprocess the data
def preprocess_data(data):
    data = data.transpose()
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data.ffill().bfill())
    return normalized_data

# Function to apply PCA for dimensionality reduction
def apply_pca(data, n_components=10):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data)

# Function to apply K-Means clustering
def apply_kmeans(data, n_clusters=4, n_init=10):
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init)
    return kmeans.fit_predict(data)


def func_calling(print_dimension):
    # Downloading the stock data
    stock_data = download_stock_data(start_date, end_date)

    # preprocessing the data
    normalized_data = preprocess_data(stock_data)

    # Applying PCA
    reduced_data = apply_pca(normalized_data, n_components=10)
    #reduced_data = apply_pca(normalized_data)

    #################### checking the rows of each data
    def print_dimensions():
        # Print the dimensions of each stock after PCA
        for i, ticker in enumerate(stock_data.columns):
            num_rows_original = stock_data[ticker].shape[0]
            num_columns_original = 1  # Assuming each stock is represented as a single column
            num_rows_pca, num_columns_pca = reduced_data.shape
            print(f"Stock {ticker} has {num_rows_original} rows and {num_columns_original} column(s) before PCA")
            print(f"Stock {ticker} has {num_rows_pca} rows and {num_columns_pca} columns after PCA")

    if print_dimension == True:
        print_dimensions()
    ## only call the function if it is true

    #################### checking the rows of each data

    # Applying K-Means Clustering
    clusters = apply_kmeans(reduced_data)

    # Grouping stocks into clusters
    valid_tickers = stock_data.columns.tolist()
    stock_clusters = pd.DataFrame({'Ticker': valid_tickers, 'Cluster': clusters})

    # Printing the clusters
    for i in range(4):
        print(f"Cluster {i}:")
        print(stock_clusters[stock_clusters['Cluster'] == i]['Ticker'].tolist(), "\n")

    # returning the clusters
    cluster_list = []
    for i in range(4):
        cluster_list.append(stock_clusters[stock_clusters['Cluster'] == i]['Ticker'].tolist())

    return cluster_list
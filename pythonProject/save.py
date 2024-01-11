# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
##################################################################################
##### danny
nasdaq_100_tickers = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "NVDA", "PYPL", "NFLX",
    "INTC", "CSCO", "ADBE", "PEP", "CMCSA", "AVGO", "AMGN", "QCOM", "TXN", "SBUX",
    "COST", "TMUS", "ISRG", "REGN", "MRNA", "JD", "ASML", "AMD", "GILD",
    "CHTR", "INTU", "CSX", "MU", "ADI", "AMAT", "MELI", "BKNG", "LRCX",
    "ZM", "ROST", "MDLZ", "KHC", "IDXX", "ILMN", "EA", "AEP", "KDP", "NXPI", "DOCU",
    "KMX", "MNST", "PAYX", "ALGN", "CTAS", "BIDU", "EXC", "KLAC", "LULU",
    "WDC", "VRTX", "CPRT", "WDAY", "ROKU", "VRSK", "SNPS", "ADI", "REG",
    "VRSN", "XEL", "ALB", "CDW", "CDNS", "FOX", "WBA", "ANSS", "SWKS",
    "XRAY", "CTSH", "DLTR", "ULTA", "TTWO", "NTES", "NTAP", "CPRT", "MCHP",
    "MTCH", "CDW", "LBTYK", "ORLY", "SPLK", "FTNT", "OKTA", "SGEN", "NXST",
    "CDW", "WYNN", "IDXX", "NTAP", "QRVO", "PDD", "FIS",
]
# from  wiki:
[
    'ADBE', 'ADP', 'ABNB', 'GOOGL', 'GOOG', 'AMZN', 'AMD', 'AEP', 'AMGN', 'ADI',
    'ANSS', 'AAPL', 'AMAT', 'ASML', 'AZN', 'TEAM', 'ADSK', 'BKR', 'BIIB', 'BKNG',
    'AVGO', 'CDNS', 'CDW', 'CHTR', 'CTAS', 'CSCO', 'CCEP', 'CTSH', 'CMCSA', 'CEG',
    'CPRT', 'CSGP', 'COST', 'CRWD', 'CSX', 'DDOG', 'DXCM', 'FANG', 'DLTR', 'DASH',
    'EA', 'EXC', 'FAST', 'FTNT', 'GEHC', 'GILD', 'GFS', 'HON', 'IDXX', 'ILMN',
    'INTC', 'INTU', 'ISRG', 'KDP', 'KLAC', 'KHC', 'LRCX', 'LULU', 'MAR', 'MRVL',
    'MELI', 'META', 'MCHP', 'MU', 'MSFT', 'MRNA', 'MDLZ', 'MDB', 'MNST', 'NFLX',
    'NVDA', 'NXPI', 'ORLY', 'ODFL', 'ON', 'PCAR', 'PANW', 'PAYX', 'PYPL', 'PDD',
    'PEP', 'QCOM', 'REGN', 'ROP', 'ROST', 'SIRI', 'SPLK', 'SBUX', 'SNPS', 'TTWO',
     'TSLA', 'TXN', 'TTD', 'VRSK', 'VRTX', 'WBA', 'WBD', 'WDAY', 'XEL', 'ZS'
]


## Download 1-year data for each company
# Download 1-year data for each company
def download_stock_data(tickers):
    stock_data = {}
    for ticker in tickers:
        try:
            data = yf.download(ticker, period="1y", interval="1d")
            if not data.empty:
                stock_data[ticker] = data['Close']   ## the attribute of the data
        except Exception as e:
            print(f"Error downloading data for {ticker}: {e}")
    return pd.DataFrame(stock_data)


# Preprocess the data
def preprocess_data(data):
    # Transpose the data so that stocks are rows and dates are columns
    data = data.transpose()
    # Normalize the data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data.ffill().bfill())  # Fill forward and backward
    return normalized_data

# Apply PCA for dimensionality reduction
def apply_pca(data, n_components=10):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data)


# Apply K-Means clustering
def apply_kmeans(data, n_clusters=4, n_init=10):
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init)
    return kmeans.fit_predict(data)


# Downloading the stock data
stock_data = download_stock_data(nasdaq_100_tickers)
print(stock_data.head())

# Filter out tickers that didn't have data
valid_tickers = stock_data.columns.tolist()
print(len(valid_tickers))


# Preprocessing the data
normalized_data = preprocess_data(stock_data)

# Applying PCA
reduced_data = apply_pca(normalized_data)

# Applying K-Means Clustering
clusters = apply_kmeans(reduced_data)

# Grouping stocks into clusters
stock_clusters = pd.DataFrame({'Ticker': valid_tickers, 'Cluster': clusters})

# Selecting one representative stock from each cluster
representative_stocks = stock_clusters.groupby('Cluster').first().reset_index()

print(representative_stocks)

# You can now analyze these representa


#################### this is a funciton in the main calling the nasdaq100
## date variables  for the stock retrieval!
end_date = datetime.now()
start_date = end_date - timedelta(days=365)
def clustering_100_companies():
    # Downloading the stock data
    from nasdaq_100 import download_stock_data
    stock_data = download_stock_data( start_date, end_date)

    # preprocessing the data
    from nasdaq_100 import preprocess_data
    normalized_data = preprocess_data(stock_data)

    # Applying PCA
    from nasdaq_100 import apply_pca
    reduced_data = apply_pca(normalized_data)

    # Applying K-Means Clustering
    from nasdaq_100 import apply_kmeans
    clusters = apply_kmeans(reduced_data)

    # Grouping stocks into clusters
    valid_tickers = stock_data.columns.tolist()
    stock_clusters = pd.DataFrame({'Ticker': valid_tickers, 'Cluster': clusters})

    # Printing the clusters
    for i in range(4):
        print(f"Cluster {i}:")
        print(stock_clusters[stock_clusters['Cluster'] == i]['Ticker'].tolist(), "\n")

 #################### this is a funciton in the main calling the nasdaq100
















def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

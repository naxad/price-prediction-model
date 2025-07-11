import requests
import pandas as pd

# Set your Alpha Vantage API Key
ALPHA_VANTAGE_API_KEY = "2I6HFTBOBR0WTH5W"

def fetch_historical_stock_data(symbol, outputsize="full"):
    """
    Fetch historical stock data using the Alpha Vantage API.

    Args:
        symbol (str): The stock ticker symbol (e.g., 'AAPL').
        outputsize (str): The size of the data ('compact' for 100 days or 'full' for all data).

    Returns:
        pd.DataFrame: A DataFrame containing stock data.
    """
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "apikey": ALPHA_VANTAGE_API_KEY,
        "outputsize": outputsize
    }
    response = requests.get(url, params=params)
    data = response.json()

    # Check for errors in the response
    if "Time Series (Daily)" not in data:
        raise ValueError("Error fetching data. Check your API key and symbol.")

    # Process the response into a DataFrame
    time_series = data["Time Series (Daily)"]
    df = pd.DataFrame.from_dict(time_series, orient="index", dtype=float)
    df = df.rename(columns={
        "1. open": "Open",
        "2. high": "High",
        "3. low": "Low",
        "4. close": "Close",
        "6. volume": "Volume"
    })
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

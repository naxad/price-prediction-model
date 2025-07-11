import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from hmmlearn.hmm import GaussianHMM

def get_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    return data[['Open', 'High', 'Low', 'Close', 'Volume']]

def preprocess_data(data, feature_columns):
    """
    Scales the selected features using MinMaxScaler.
    """
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[feature_columns])
    return data_scaled, scaler, len(feature_columns)

def create_sequences(data, sequence_length):
    """
    Create sequences from the scaled data.
    """
    X, y = [], []
    num_features = data.shape[1]
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length, 0])  # Assuming target is the 'Close' price.
    return np.array(X).reshape(-1, sequence_length, num_features), np.array(y)

def compute_hmm_regimes(data, n_states=3):
    """
    Compute market regimes using a Hidden Markov Model (HMM) on log returns.
    """
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    data.dropna(inplace=True)
    hmm = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1000)
    hmm.fit(data[['Log_Returns']])
    data['Market_Regime'] = hmm.predict(data[['Log_Returns']])
    return data

import talib
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import numpy as np

def compute_ta_indicators(data):
    """
    Compute multiple TA indicators using TA-Lib and add them as features.
    """
    data['SMA_20'] = talib.SMA(data['Close'], timeperiod=20)
    data['SMA_50'] = talib.SMA(data['Close'], timeperiod=50)
    data['EMA_20'] = talib.EMA(data['Close'], timeperiod=20)
    data['EMA_50'] = talib.EMA(data['Close'], timeperiod=50)

    data['RSI_14'] = talib.RSI(data['Close'], timeperiod=14)
    data['MACD'], data['MACD_signal'], _ = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

    if 'Volume' in data.columns:
        data['MFI'] = talib.MFI(data['High'], data['Low'], data['Close'], data['Volume'], timeperiod=14)
    else:
        print("⚠ Warning: 'Volume' data missing! MFI cannot be computed.")

    data['ATR_14'] = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)
    data['Bollinger_Upper'], data['Bollinger_Middle'], data['Bollinger_Lower'] = talib.BBANDS(
        data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    data['ADX_14'] = talib.ADX(data['High'], data['Low'], data['Close'], timeperiod=14)

    # Additional indicators (optional):
    data['CCI'] = talib.CCI(data['High'], data['Low'], data['Close'], timeperiod=14)
    data['STOCH_K'], data['STOCH_D'] = talib.STOCH(data['High'], data['Low'], data['Close'])
    data['Williams_%R'] = talib.WILLR(data['High'], data['Low'], data['Close'], timeperiod=14)

    print("✅ Available Features in Data:", data.columns.tolist())
    data.dropna(inplace=True)
    return data

def compute_market_volatility(data, window=20):
    """
    Compute rolling volatility as the standard deviation of daily returns.
    """
    data["Market_Volatility"] = data["Close"].pct_change().rolling(window=window).std()
    return data

def update_ta_features(last_row, predicted_price):
    """
    Dynamically update key TA columns based on the new predicted price.
    Simplified approach: weighted averages for SMAs, exponential for EMAs, etc.
    """
    new_row = last_row.copy()
    for col in ["Open", "High", "Low", "Close"]:
        new_row[col] = predicted_price

    # Weighted update for SMAs
    if "SMA_20" in last_row:
        new_row["SMA_20"] = (predicted_price + 19 * last_row["SMA_20"]) / 20
    if "SMA_50" in last_row:
        new_row["SMA_50"] = (predicted_price + 49 * last_row["SMA_50"]) / 50

    # Exponential update for EMAs
    if "EMA_20" in last_row:
        alpha = 2 / (20 + 1)
        new_row["EMA_20"] = predicted_price * alpha + last_row["EMA_20"] * (1 - alpha)
    if "EMA_50" in last_row:
        alpha = 2 / (50 + 1)
        new_row["EMA_50"] = predicted_price * alpha + last_row["EMA_50"] * (1 - alpha)

    return new_row

def compute_hmm_sentiment(data, period=14, n_states=2, eps=1e-5, aggregator="ma"):
    """
    Compute an HMM-based sentiment signal from daily returns.
    Using n_states=2 by default and aggregator='ma' for a rolling average.
    """
    # 1. Daily returns
    data['Return'] = data['Close'].pct_change()
    data = data.dropna().copy()

    # 2. Fit HMM
    returns = data['Return'].values.reshape(-1, 1)
    hmm = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1000)
    hmm.fit(returns)
    hidden_states = hmm.predict(returns)
    data['HMM_State'] = hidden_states

    # 3. State => daily sentiment
    state_mean = {}
    for s in range(n_states):
        mean_val = data.loc[data['HMM_State'] == s, 'Return'].mean()
        state_mean[s] = mean_val

    state_sentiment = {}
    for s in range(n_states):
        val = state_mean[s]
        if val > eps:
            state_sentiment[s] = 1
        elif val < -eps:
            state_sentiment[s] = -1
        else:
            state_sentiment[s] = 0

    data['HMM_Daily_Sentiment'] = data['HMM_State'].map(state_sentiment)

    # 4. Rolling aggregator
    if aggregator.lower() == "sum":
        data['HMM_Cum_Sentiment'] = data['HMM_Daily_Sentiment'].rolling(window=period).sum()
    elif aggregator.lower() == "ma":
        data['HMM_Cum_Sentiment'] = data['HMM_Daily_Sentiment'].rolling(window=period).mean()
    else:
        data['HMM_Cum_Sentiment'] = data['HMM_Daily_Sentiment'].rolling(window=period).sum()

    return data, hmm, state_sentiment

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import timedelta, date
from data_loader import get_stock_data, preprocess_data, create_sequences
from technical_indicators import (
    compute_ta_indicators,
    compute_market_volatility,
    update_ta_features,
    compute_hmm_sentiment
)
from news_sentiment import compute_news_sentiment
from models_tft import build_tft_model
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import plotly.graph_objects as go

# Set seeds for reproducibility.
import random
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Ensure GPU memory growth.
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Define feature set (including HMM cumulative sentiment).
feature_columns = [
    "Open", "High", "Low", "Close", "Volume",
    "News_Sentiment", "Market_Volatility",
    "SMA_20", "SMA_50", "EMA_20", "EMA_50",
    "RSI_14", "MACD", "MFI", "ATR_14",
    "Bollinger_Upper", "Bollinger_Middle", "Bollinger_Lower", "ADX_14",
    "HMM_Cum_Sentiment"
]

def inverse_transform_price(scaled_price, scaler, close_index):
    return scaled_price * scaler.data_range_[close_index] + scaler.data_min_[close_index]

if __name__ == "__main__":
    ticker = "AAPL"
    start_date = "2005-01-01"
    end_date = date.today().strftime('%Y-%m-%d')
    
    # 1. Fetch and process data.
    data = get_stock_data(ticker, start_date, end_date)
    data = compute_news_sentiment(data, keyword="Apple Stock")
    data = compute_market_volatility(data)
    data = compute_ta_indicators(data)
    data, hmm_model, state_sentiment = compute_hmm_sentiment(data, period=14)
    data.dropna(inplace=True)
    
    # 2. Prepare training data.
    sequence_length = 60
    scaled_data, scaler, num_features = preprocess_data(data, feature_columns)
    X, y_price = create_sequences(scaled_data, sequence_length)
    # (For the TFT, we use only the price output for now; regime target can be added similarly.)
    
    # Use an 80/20 split.
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_price_train, y_price_val = y_price[:train_size], y_price[train_size:]
    
    # 3. Build and train the TFT model.
    input_shape = (sequence_length, num_features)
    quantiles = [0.1, 0.5, 0.9]
    num_regimes = 3  # For regime classification.
    tft_model = build_tft_model(input_shape, quantiles=quantiles, num_regimes=num_regimes)
    tft_model.summary()
    
    callbacks = [EarlyStopping(monitor='val_loss', patience=10),
                 ReduceLROnPlateau(monitor='val_loss', patience=5)]
    
    # For simplicity, we use the price target as y_price (reshaped to (-1,1)) and dummy regime labels.
    # In practice, you would prepare regime targets (one-hot encoded) from your HMM sentiment.
    dummy_regime_train = np.zeros((y_price_train.shape[0], num_regimes))
    dummy_regime_train[:,1] = 1  # Assume neutral (index 1) for training.
    dummy_regime_val = np.zeros((y_price_val.shape[0], num_regimes))
    dummy_regime_val[:,1] = 1

    history = tft_model.fit(
        X_train,
        {"price_quantiles": y_price_train.reshape(-1, 1), "regime_class": dummy_regime_train},
        validation_data=(X_val, {"price_quantiles": y_price_val.reshape(-1, 1), "regime_class": dummy_regime_val}),
        epochs=50,
        batch_size=32,
        callbacks=callbacks
    )
    
    # 4. Iterative forecasting.
    forecast_steps = 25
    future_dates = []
    price_preds = []
    regime_preds = []
    current_date = data.index[-1]
    
    # Use the last sequence in X as the initial input.
    input_seq = np.expand_dims(X[-1], axis=0)  # shape: (1, sequence_length, num_features)
    
    for i in range(forecast_steps):
        preds = tft_model.predict(input_seq)
        # For price, take the median quantile (index 1).
        forecast_price_scaled = preds[0][0, 1]
        close_idx = feature_columns.index("Close")
        forecast_price = inverse_transform_price(forecast_price_scaled, scaler, close_idx)
        price_preds.append(forecast_price)
        
        # For regime, take argmax (dummy in this example).
        regime_prob = preds[1][0]
        regime_class = np.argmax(regime_prob)
        regime_preds.append(regime_class)
        
        # Update the input sequence.
        # Here, we simulate a new row using update_ta_features.
        # new_row_scaled should be reshaped to (1, 1, num_features) to match input_seq.
        last_row_raw = data.iloc[-1][feature_columns].copy()  # You may want a more dynamic update.
        updated_row = update_ta_features(last_row_raw, forecast_price)
        new_row_scaled = scaler.transform([updated_row.values])[0]
        # Fix: reshape new_row_scaled to (1,1,num_features)
        new_row_scaled = new_row_scaled.reshape(1, 1, -1)
        input_seq = np.concatenate([input_seq[:, 1:, :], new_row_scaled], axis=1)
        
        current_date += timedelta(days=1)
        while current_date.weekday() >= 5:
            current_date += timedelta(days=1)
        future_dates.append(current_date)
    
    # 5. Plot results.
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data["Close"].values,
        mode='lines',
        name='Actual Price',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=price_preds,
        mode='lines',
        name='TFT Forecast (Median)',
        line=dict(color='red', dash='dot')
    ))
    fig.update_layout(
        title="TFT Multi-task Forecasting",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white"
    )
    fig.show()

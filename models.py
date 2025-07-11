from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, Input, MultiHeadAttention,
    LayerNormalization, Conv1D, Flatten
)
from tensorflow.keras.optimizers import Adam
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

def train_hmm(data, n_states=3):
    """
    Trains a Gaussian HMM for an old version; not used directly if aggregator is set in compute_hmm_sentiment.
    """
    hmm = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1000)
    hmm.fit(data)
    hidden_states = hmm.predict(data)
    return hmm, hidden_states

def transformer_encoder(inputs, head_size=64, num_heads=4, ff_dim=64, dropout=0.2):
    """
    Simple transformer encoder block used in the hybrid model.
    """
    x = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs  # Residual
    x = Dense(ff_dim, activation="relu")(res)
    x = Dense(inputs.shape[-1])(x)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    return x + res

def build_cnn_model(input_shape):
    """
    A CNN model for sequence data:
    - Slightly smaller learning rate
    - Two Conv1D layers with moderate dropout
    """
    model = Sequential([
        Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.2),
        Conv1D(filters=64, kernel_size=5, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss="huber")
    return model

def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(encoded)
    output_layer = Dense(input_dim, activation='linear')(decoded)
    autoencoder = Model(input_layer, output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def build_hybrid_model(input_shape):
    """
    Hybrid model with:
    - 2 LSTM layers
    - Transformer block in-between
    - A small CNN block
    """
    inputs = Input(shape=input_shape)
    # LSTM #1
    x = LSTM(256, return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Transformer block
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.2)
    
    # CNN block
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    
    # LSTM #2
    x = LSTM(128, return_sequences=False)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    
    # Dense
    x = Dense(64, activation="relu")(x)
    outputs = Dense(1, activation="linear")(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss="huber")
    return model

def train_xgboost(X_train, y_train):
    """
    XGBoost with more trees and smaller LR => can help with non-linear patterns
    """
    model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=500,
        learning_rate=0.01
    )
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """
    A moderately larger forest for better smoothing of the predictions
    """
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_regression(X_train, y_train):
    """
    Simple linear regression for baseline/trend correction.
    """
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    y_train_flat = y_train.reshape(-1, 1)
    reg_model = LinearRegression()
    reg_model.fit(X_train_flat, y_train_flat)
    return reg_model

def train_xgboost_ta(data):
    """
    XGBoost with only TA features => for next-day close
    """
    ta_features = [
        "SMA_20", "SMA_50", "EMA_20", "EMA_50", 
        "RSI_14", "MACD", "MFI", "ATR_14",
        "Bollinger_Upper", "Bollinger_Middle", "Bollinger_Lower", "ADX_14"
    ]
    data = data.dropna()
    X = data[ta_features]
    y = data["Close"].shift(-1)
    X_train, X_test, y_train, y_test = train_test_split(X[:-1], y[:-1], test_size=0.2, shuffle=False)
    model = XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.01)
    model.fit(X_train, y_train)
    return model

def train_future_prediction_model(data):
    """
    A separate model for 30-day-ahead (5-day rolling) predictions
    """
    feature_columns = [
        "Open", "High", "Low", "Close", "Volume",
        "SMA_20", "SMA_50", "EMA_20", "EMA_50", 
        "RSI_14", "MACD", "MFI", "ATR_14", 
        "Bollinger_Upper", "Bollinger_Middle", "Bollinger_Lower", "ADX_14",
        "News_Sentiment", "Market_Volatility"
    ]
    X = data[feature_columns].values
    y = data["Close"].shift(-30).rolling(5).mean().dropna().values
    X = X[:len(y)]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = XGBRegressor(n_estimators=250, learning_rate=0.025)
    model.fit(X_train, y_train)
    return model

# --------------------------------------------------------------------------
# NEW: a small neural network ensemble (instead of simple LinearRegression)
# --------------------------------------------------------------------------
def build_ensemble_neural_net():
    """
    A small feed-forward net that takes 4 inputs (the 4 base predictions)
    and outputs a single blended prediction (scaled).
    """
    model = Sequential([
        Dense(16, activation='relu', input_shape=(4,)),
        Dropout(0.1),
        Dense(8, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model

def train_ensemble_model(hybrid_model, cnn_model, xgb_model, rf_model, X_val, y_val, sequence_length, feature_columns):
    """
    Replaces the simple linear meta-model with a small neural net ensemble.
    """
    # Predictions from each base model
    hybrid_preds = hybrid_model.predict(X_val).flatten()
    cnn_preds   = cnn_model.predict(X_val).flatten()
    flat_features = X_val.reshape(X_val.shape[0], -1)
    xgb_preds   = xgb_model.predict(flat_features)
    rf_preds    = rf_model.predict(flat_features)

    # Stack them
    ensemble_inputs = np.column_stack((hybrid_preds, cnn_preds, xgb_preds, rf_preds))
    
    # Train a small neural net
    ensemble_net = build_ensemble_neural_net()
    ensemble_net.fit(
        ensemble_inputs, y_val,
        batch_size=32, epochs=100,
        verbose=1,
        validation_split=0.1
    )
    return ensemble_net

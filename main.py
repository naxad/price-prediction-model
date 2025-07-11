import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import timedelta, date
import logging
import sys

# -----------------------------
# Configure Logging
# -----------------------------
LOG_FILENAME = 'debug_log.txt'
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILENAME, mode='w'),   # Log to file (overwrite mode)
        logging.StreamHandler(sys.stdout)              # Also log to console
    ]
)

from data_loader import get_stock_data, preprocess_data, create_sequences
from technical_indicators import (
    compute_ta_indicators,
    compute_market_volatility,
    update_ta_features,
    compute_hmm_sentiment
)
from news_sentiment import compute_news_sentiment
from models import (
    build_hybrid_model,
    build_cnn_model,
    train_regression,
    train_xgboost,
    train_random_forest,
    train_future_prediction_model,
    train_ensemble_model  # Now returns a small NN for the ensemble
)
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from training_memory import load_trained_model, get_callbacks, save_trained_model
import plotly.graph_objects as go

# --------------------------------------------------
# GPU Configuration
# --------------------------------------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
logging.info("GPU configured (if available).")

# --------------------------------------------------
# Define Feature Set
# --------------------------------------------------
feature_columns = [
    "Open", "High", "Low", "Close", "Volume",
    "News_Sentiment",
    "Market_Volatility",
    "SMA_20", "SMA_50", "EMA_20", "EMA_50",
    "RSI_14", "MACD", "MFI", "ATR_14",
    "Bollinger_Upper", "Bollinger_Middle", "Bollinger_Lower", "ADX_14",
    "HMM_Cum_Sentiment"
]
logging.info(f"Feature columns: {feature_columns}")

def inverse_transform_price(scaled_price, scaler, close_index):
    """
    Inverse-transform a single scaled price value using the MinMaxScaler parameters.
    """
    return scaled_price * scaler.data_range_[close_index] + scaler.data_min_[close_index]

def predict_future_prices_ensemble(
    ensemble_model, models_dict, data, scaler, sequence_length, days_into_future=30
):
    """
    Predict future prices using an ensemble neural net meta-model that blends
    the 4 base predictions each day. Dynamically updates the TA row.

    models_dict => {"hybrid": hybrid_model, "cnn": cnn_model, "xgb": xgb_model, "rf": rf_model}
    """
    close_index = feature_columns.index("Close")

    # Debug tail of data
    logging.debug("Tail of entire data:\n%s", data.tail(5))
    logging.debug("Data final row => Close=%.4f, date=%s", data["Close"].iloc[-1], data.index[-1])

    # Last 60 rows that truly end on final day
    last_sequence_raw = data.iloc[-sequence_length:][feature_columns]
    logging.debug(
        "DEBUG: After slicing last_sequence_raw => final row Close=%.4f, date=%s",
        last_sequence_raw["Close"].iloc[-1],
        last_sequence_raw.index[-1]
    )
    last_sequence_scaled = scaler.transform(last_sequence_raw.values)

    # Keep track of last row
    last_row_raw = last_sequence_raw.iloc[-1].copy()

    # Initial predicted price = final day’s actual close
    first_predicted_price = last_sequence_raw["Close"].iloc[-1]
    predictions = [first_predicted_price]
    current_date = last_sequence_raw.index[-1]
    future_dates = [current_date]

    logging.debug(
        "DEBUG: Setting initial predicted price to final row’s Close=%.4f on date=%s",
        first_predicted_price, current_date
    )

    for day_ahead in range(1, days_into_future):
        seq_input = last_sequence_scaled.reshape(1, sequence_length, len(feature_columns))
        flat_input = last_sequence_scaled.reshape(1, -1)

        # Base predictions
        hybrid_pred = models_dict["hybrid"].predict(seq_input)[0][0]
        cnn_pred    = models_dict["cnn"].predict(seq_input)[0][0]
        xgb_pred    = models_dict["xgb"].predict(flat_input)[0]
        rf_pred     = models_dict["rf"].predict(flat_input)[0]

        logging.debug(
            "Day %d: Hybrid=%.4f, CNN=%.4f, XGB=%.4f, RF=%.4f",
            day_ahead, hybrid_pred, cnn_pred, xgb_pred, rf_pred
        )

        # The ensemble model is now a small neural net
        # => feed [hybrid_pred, cnn_pred, xgb_pred, rf_pred]
        ensemble_input = np.array([[hybrid_pred, cnn_pred, xgb_pred, rf_pred]])
        scaled_pred_price = ensemble_model.predict(ensemble_input)[0][0]
        predicted_price = inverse_transform_price(scaled_pred_price, scaler, close_index)
        predictions.append(predicted_price)

        # Advance date (skip weekends)
        current_date += timedelta(days=1)
        while current_date.weekday() >= 5:
            current_date += timedelta(days=1)
        future_dates.append(current_date)

        logging.debug(
            "Day %d => predicted price=%.4f for %s",
            day_ahead, predicted_price, current_date
        )

        # Update the last row’s TA
        updated_row_raw = update_ta_features(last_row_raw, predicted_price)
        new_row_scaled = scaler.transform([updated_row_raw.values])[0]
        last_sequence_scaled = np.vstack([last_sequence_scaled[1:], new_row_scaled])
        last_row_raw = updated_row_raw.copy()

    return future_dates, predictions

if __name__ == "__main__":
    ticker = "AAPL"
    start_date = "2005-01-01"
    end_date = date.today().strftime('%Y-%m-%d')

    logging.info("Fetching stock data...")
    data = get_stock_data(ticker, start_date, end_date)
    logging.info("Data shape: %s", data.shape)

    # Compute news sentiment
    logging.info("Computing news sentiment...")
    data = compute_news_sentiment(data, keyword="Apple Stock")
    logging.debug("Sample News Sentiment:\n%s", data["News_Sentiment"].head())

    # TA indicators & volatility
    data = compute_market_volatility(data)
    data = compute_ta_indicators(data)

    # HMM sentiment
    logging.info("Computing HMM sentiment (n_states=2, aggregator='ma')...")
    data, hmm_model, state_sentiment = compute_hmm_sentiment(
        data, period=14, n_states=2, aggregator="ma"
    )
    logging.debug("HMM Cum Sentiment sample:\n%s", data["HMM_Cum_Sentiment"].head(10))

    data.dropna(inplace=True)
    logging.info("Data shape after dropna: %s", data.shape)

    # Prepare training sequences
    sequence_length = 60
    scaled_data, scaler, num_features = preprocess_data(data, feature_columns)
    X, y = create_sequences(scaled_data, sequence_length)
    logging.info("X shape: %s, y shape: %s", X.shape, y.shape)

    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    logging.info("Train size: %s, Val size: %s", X_train.shape, X_val.shape)

    # Train baseline & trees
    logging.info("Training regression model...")
    reg_model = train_regression(X_train.reshape(X_train.shape[0], -1), y_train)
    logging.info("Training XGBoost...")
    xgb_model = train_xgboost(X_train.reshape(X_train.shape[0], -1), y_train)
    logging.info("Training RandomForest...")
    rf_model = train_random_forest(X_train.reshape(X_train.shape[0], -1), y_train)

    # CNN
    logging.info("Building & training CNN model...")
    cnn_model = build_cnn_model((sequence_length, X_train.shape[-1]))
    cnn_model.fit(
        X_train, y_train,
        batch_size=32, epochs=100,
        validation_data=(X_val, y_val),
        verbose=1,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', patience=5)
        ]
    )

    # Hybrid
    logging.info("Building & training Hybrid model...")
    loaded_model, saved_num_features = load_trained_model()
    if loaded_model and saved_num_features == num_features:
        hybrid_model = loaded_model
    else:
        hybrid_model = build_hybrid_model((sequence_length, X_train.shape[-1]))

    hybrid_model.fit(
        X_train, y_train,
        batch_size=32, epochs=75,
        validation_data=(X_val, y_val),
        callbacks=get_callbacks(),
        verbose=1
    )
    save_trained_model(hybrid_model, num_features)

    # Future Price Model
    logging.info("Training Future Price Prediction Model...")
    future_model = train_future_prediction_model(data)

    # Ensemble (small neural net)
    logging.info("Training ensemble meta-model (neural net)...")
    ensemble_model = train_ensemble_model(
        hybrid_model, cnn_model, xgb_model, rf_model,
        X_val, y_val, sequence_length, feature_columns
    )

    # Validation predictions (hybrid)
    logging.info("Checking sample predictions from Hybrid on validation set...")
    hybrid_val_preds_scaled = hybrid_model.predict(X_val).flatten()
    close_index = feature_columns.index("Close")
    hybrid_val_preds = np.array([
        inverse_transform_price(p, scaler, close_index) 
        for p in hybrid_val_preds_scaled
    ])
    logging.debug("Sample Hybrid Predictions after inverse scaling: %s", hybrid_val_preds[:5])

    # Future predictions
    logging.info("Forecasting future prices with the ensemble meta-model...")
    models_dict = {"hybrid": hybrid_model, "cnn": cnn_model, "xgb": xgb_model, "rf": rf_model}
    future_days = 25
    future_dates, future_predictions = predict_future_prices_ensemble(
        ensemble_model, models_dict, data, scaler, sequence_length, days_into_future=future_days
    )
    logging.debug("Future predictions (first 5): %s", future_predictions[:5])

    # Construct final array
    adjusted_future_predictions = future_predictions  # no extra factor/hunch

    # Market outlook from HMM
    if "HMM_Cum_Sentiment" in data.columns:
        latest_sentiment = data["HMM_Cum_Sentiment"].iloc[-1]
    else:
        latest_sentiment = 0.0

    if latest_sentiment > 0:
        sentiment_message = f"Bullish Trend Detected (MA Score: {latest_sentiment})."
    elif latest_sentiment < 0:
        sentiment_message = f"Bearish Trend Detected (MA Score: {latest_sentiment})."
    else:
        sentiment_message = f"Neutral Trend Detected (MA Score: {latest_sentiment})."

    logging.info("Perform feature importance analysis if needed (SHAP, meta-model weights, etc.)")

    # Plot results
    logging.info("Plotting final results...")
    fig = go.Figure()
    # Actual (last chunk)
    fig.add_trace(go.Scatter(
        x=data.index[-len(y_val):],
        y=data["Close"].values[-len(y_val):],
        mode='lines', name='Actual Price', line=dict(color='blue')
    ))
    # Hybrid val
    fig.add_trace(go.Scatter(
        x=data.index[-len(y_val):],
        y=hybrid_val_preds,
        mode='lines', name='Hybrid Prediction', line=dict(dash='dash', color='orange')
    ))
    # Ensemble future
    fig.add_trace(go.Scatter(
        x=future_dates, y=adjusted_future_predictions,
        mode='lines', name="Future Predictions", line=dict(dash='dot', color='green')
    ))
    fig.update_layout(
        title=f"Stock Price Prediction with Revised HMM & Debug\n{sentiment_message}",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        hovermode="x"
    )
    fig.show()

    logging.info("Completed main.py execution. Check '%s' for full debug logs.", LOG_FILENAME)

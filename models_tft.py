import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Flatten, MultiHeadAttention, Add
from tensorflow.keras.optimizers import Adam

def quantile_loss(q):
    """Return a function that computes quantile loss for quantile q."""
    def loss(y_true, y_pred):
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(q * e, (q - 1) * e))
    return loss

def combined_quantile_loss(quantiles):
    """A loss function that averages losses over multiple quantiles."""
    def loss(y_true, y_pred):
        losses = []
        for i, q in enumerate(quantiles):
            losses.append(quantile_loss(q)(y_true, y_pred[:, i]))
        return tf.reduce_mean(tf.stack(losses))
    return loss

def build_tft_model(input_shape, quantiles=[0.1, 0.5, 0.9], num_regimes=3):
    """
    Build a simplified Temporal Fusion Transformer–inspired multi-task model.
    
    The model has two heads:
      - "price_quantiles": Outputs one value per quantile (e.g. 0.1, 0.5, 0.9).
      - "regime_class": A softmax output for market regime classification.
    """
    inputs = Input(shape=input_shape)  # (T, num_features)
    
    # A simple dense layer for feature encoding (simulating variable selection)
    x = Dense(64, activation='relu')(inputs)  # (T, 64)
    
    # LSTM layer to capture temporal dependencies.
    lstm_out = LSTM(258, return_sequences=True)(x)  # (T, 128)
    
    # Multi-head self-attention to capture long-range interactions.
    attn_out = MultiHeadAttention(num_heads=4, key_dim=32)(lstm_out, lstm_out)
    
    # Residual connection.
    x_res = Add()([lstm_out, attn_out])
    x_res = Dense(128, activation='relu')(x_res)
    
    # Flatten the time dimension.
    flat = Flatten()(x_res)
    flat = Dropout(0.1)(flat)
    
    # Shared representation.
    shared = Dense(128, activation='relu')(flat)
    
    # Price quantile regression head.
    price_output = Dense(len(quantiles), activation='linear', name="price_quantiles")(shared)
    
    # Regime classification head.
    regime_output = Dense(num_regimes, activation='softmax', name="regime_class")(shared)
    
    model = Model(inputs=inputs, outputs=[price_output, regime_output])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={
            "price_quantiles": combined_quantile_loss(quantiles),
            "regime_class": "categorical_crossentropy"
        },
        loss_weights={"price_quantiles": 1.0, "regime_class": 0.5}
    )
    
    return model

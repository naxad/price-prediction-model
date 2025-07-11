import os
import tensorflow as tf
import pandas as pd
import pickle

BEST_MODEL_PATH = "best_model.h5"
HISTORY_PATH = "training_history.csv"
FEATURES_PATH = "num_features.pkl"  # New file to store number of features

def save_training_history(y_true, y_pred):
    """
    Save the training history (actual vs predicted prices) for tracking.
    """
    # Ensure arrays have the same length
    min_length = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:min_length], y_pred[:min_length]

    df = pd.DataFrame({'Actual Price': y_true, 'Predicted Price': y_pred})
    df.to_csv(HISTORY_PATH, index=False)
    print(f"✅ Training history saved to {HISTORY_PATH}")


def save_trained_model(model, num_features):
    """
    Save the trained model along with its expected number of features.
    """
    model.save(BEST_MODEL_PATH)
    
    # Save the number of features
    with open(FEATURES_PATH, "wb") as f:
        pickle.dump(num_features, f)

    print(f"✅ Model saved with {num_features} features.")

def load_trained_model():
    """
    Load the best saved model if available, ensuring feature compatibility.
    """
    if os.path.exists(BEST_MODEL_PATH) and os.path.exists(FEATURES_PATH):
        print("📥 Loading best saved model...")

        model = tf.keras.models.load_model(BEST_MODEL_PATH)

        # Load number of features
        with open(FEATURES_PATH, "rb") as f:
            saved_num_features = pickle.load(f)

        print(f"🔹 Saved model expects {saved_num_features} features.")

        return model, saved_num_features
    else:
        print("⚠ No saved model found, starting fresh!")
        return None, None

def get_callbacks():
    """
    Create callbacks for training: saves the best model & tracks training progress.
    """
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        BEST_MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=25, restore_best_weights=True, verbose=1
    )

    return [checkpoint, early_stopping]

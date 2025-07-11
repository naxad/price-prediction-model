import gym
import numpy as np
import pandas as pd
import yfinance as yf
import torch
from stable_baselines3 import DQN
from trading_env import StockTradingEnv

# ✅ Step 1: Fetch Historical Data Automatically
def fetch_stock_data(ticker="AAPL", start_date="2005-01-01", end_date=None):
    """
    Fetches historical stock price data from Yahoo Finance.
    - Default ticker: AAPL
    - Default start date: 2005-01-01
    - End date: Today's date
    """
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    print(f"📡 Fetching data for {ticker} from {start_date} to {end_date}...")
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    if stock_data.empty:
        raise ValueError("❌ Failed to fetch stock data. Check ticker symbol or API limits.")
    
    stock_data.reset_index(inplace=True)
    stock_data = stock_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    # ✅ Compute technical indicators (Matching `StockTradingEnv`)
    stock_data["SMA_20"] = stock_data["Close"].rolling(window=20).mean()
    stock_data["RSI_14"] = 100 - (100 / (1 + stock_data["Close"].pct_change().rolling(14).mean()))
    stock_data["MACD"] = stock_data["Close"].ewm(span=12).mean() - stock_data["Close"].ewm(span=26).mean()

    stock_data.dropna(inplace=True)  # Remove NaN values from rolling computations
    return stock_data

# ✅ Step 2: Prepare & Normalize Data
def prepare_env(stock_data):
    """
    Converts historical stock data into a gym environment for reinforcement learning.
    """
    stock_data["Date"] = pd.to_datetime(stock_data["Date"])
    stock_data.set_index("Date", inplace=True)
    env = StockTradingEnv(stock_data)
    return env

# ✅ Step 3: Train DQN Agent with GPU Acceleration
def train_dqn(env, save_path="dqn_trader"):
    """
    Trains a Deep Q-Network (DQN) agent on the stock trading environment using GPU.
    """
    # 🔥 **Check for GPU**
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Using device: {device}")

    model = DQN(
        "MlpPolicy", 
        env, 
        learning_rate=0.0001, 
        buffer_size=10000, 
        batch_size=64, 
        learning_starts=1000, 
        target_update_interval=500, 
        train_freq=4, 
        exploration_final_eps=0.01, 
        exploration_fraction=0.1, 
        policy_kwargs={"net_arch": [256, 256]}, 
        verbose=1, 
        device=device  # ✅ NOW TRAINING ON GPU IF AVAILABLE
    )

    # ✅ Train Model
    model.learn(total_timesteps=200000)
    
    # ✅ Save Model
    model.save(save_path)
    print("✅ Model Training Complete & Saved!")

if __name__ == "__main__":
    stock_data = fetch_stock_data()
    env = prepare_env(stock_data)
    train_dqn(env)

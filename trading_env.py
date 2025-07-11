import gym
import numpy as np
import pandas as pd
from gym import spaces

class StockTradingEnv(gym.Env):
    """
    Custom Stock Trading Environment for Reinforcement Learning.
    - Observation Space: Normalized stock data with all model features.
    - Action Space: Buy, Sell, Hold.
    """
    
    def __init__(self, stock_data, window_size=60):
        super(StockTradingEnv, self).__init__()

        self.stock_data = stock_data
        self.window_size = window_size  # Context window
        self.current_step = window_size  # Start after the first `window_size` days
        
        # ✅ **Use all features from training**
        self.selected_features = stock_data.columns.tolist()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, len(self.selected_features)), dtype=np.float32
        )
        
        # ✅ **Action Space (0 = Buy, 1 = Hold, 2 = Sell)**
        self.action_space = spaces.Discrete(3)
    
    def reset(self):
        """
        Resets the environment and returns the initial observation.
        """
        self.current_step = self.window_size
        return self._get_observation()
    
    def _get_observation(self):
        """
        Returns the last `window_size` rows of selected features.
        """
        obs = self.stock_data.iloc[self.current_step - self.window_size: self.current_step].values
        return obs

    def step(self, action):
        """
        Executes the action and moves the environment forward by one step.
        """
        self.current_step += 1
        done = self.current_step >= len(self.stock_data) - 1  # Episode ends at last step
        
        # ✅ Get observation correctly (last `window_size` rows)
        obs = self._get_observation()

        # ✅ Extract the latest row for feature-based reward calculation
        latest_data = self.stock_data.iloc[self.current_step]

        # ✅ Extract individual features properly (ensuring columns exist)
        current_price = latest_data["Close"] if "Close" in latest_data else 0
        volume = latest_data["Volume"] if "Volume" in latest_data else 0
        sma = latest_data["SMA_20"] if "SMA_20" in latest_data else 0
        rsi = latest_data["RSI_14"] if "RSI_14" in latest_data else 0
        macd = latest_data["MACD"] if "MACD" in latest_data else 0

        # ✅ Reward calculation
        reward = self._compute_reward(action)

        # ✅ Return updated observation + reward
        return obs, reward, done, {"current_price": current_price, "volume": volume, "sma": sma, "rsi": rsi, "macd": macd}

    
    def _compute_reward(self, action):
        """
        Reward function based on trading action.
        """
        current_price = self.stock_data["Close"].iloc[self.current_step]
        previous_price = self.stock_data["Close"].iloc[self.current_step - 1]
        
        if action == 0:  # Buy
            return (current_price - previous_price) / previous_price
        elif action == 2:  # Sell
            return (previous_price - current_price) / previous_price
        return 0  # Hold

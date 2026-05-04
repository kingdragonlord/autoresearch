"""
Autoresearch RL pretraining script. Single-GPU, single-file.
Trains a Stable-Baselines3 SAC agent on SPY Options.
Usage: uv run train.py
"""

import os
import sys
import time
import math
import numpy as np
import pandas as pd
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from prepare import TIME_BUDGET, TICKER, LOCAL_CACHE, sync_data, evaluate_policy
from env_optiontrading_spy import OptionTradingEnv

# ---------------------------------------------------------------------------
# Training Parameters (These can be tuned by the autonomous agent)
# ---------------------------------------------------------------------------

LEARNING_RATE = 5e-5
BUFFER_SIZE = 150_000
BATCH_SIZE = 256
TAU = 0.005
GAMMA = 0.97
ENT_COEF = "auto"
NET_ARCH = [512, 512, 256]
EPISODE_LEN = 21

TECH_INDICATORS = ["open", "high", "low", "close", "volume", "macd_hist", "atr_14", "bb_width_20", "roc_21"]

# ---------------------------------------------------------------------------
# Time Limit Callback
# ---------------------------------------------------------------------------

class TimeLimitCallback(BaseCallback):
    """Stop training after a fixed time budget."""
    def __init__(self, time_budget, verbose=0):
        super().__init__(verbose)
        self.time_budget = time_budget
        self.start_time = None

    def _on_training_start(self) -> None:
        self.start_time = time.time()

    def _on_step(self) -> bool:
        elapsed = time.time() - self.start_time
        if elapsed >= self.time_budget:
            print(f"Time budget of {self.time_budget}s reached. Stopping training.")
            return False
        return True


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------

def main():
    t0_total = time.time()
    
    # 1. Sync data and load Parquet
    parquet_path = sync_data()
    if not os.path.exists(parquet_path):
        # Fallback to absolute path from scratch dir
        parquet_path = "C:/Users/cerod/.gemini/antigravity/scratch/riskpulse-backend/daily_features.parquet"
        
    df = pd.read_parquet(parquet_path)
    df = df[df["ticker"] == "SPY"].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values("date")
    df = df[df["date"] >= "2019-01-01"].copy()

    # 2. Filter dates with valid options chains
    local_files = [f for f in os.listdir(LOCAL_CACHE) if f.startswith("spy_strikes_") and f.endswith(".json")]
    valid_dates = {f.replace("spy_strikes_", "").replace(".json", "") for f in local_files}
    df_valid = df[df["date"].isin(valid_dates)].copy()

    all_dates = sorted(df_valid["date"].unique())
    split = int(len(all_dates) * 0.80)
    train_dates = set(all_dates[:split])
    val_dates = set(all_dates[split:])
    
    df_train = df_valid[df_valid["date"].isin(train_dates)].copy()
    df_val = df_valid[df_valid["date"].isin(val_dates)].copy()

    # 3. Build Environment
    def make_train_env():
        return OptionTradingEnv(
            df=df_train,
            ticker="SPY",
            initial_amount=100_000,
            options_data_path=LOCAL_CACHE,
            tech_indicator_list=TECH_INDICATORS,
            episode_len=EPISODE_LEN,
        )
        
    def make_val_env():
        return OptionTradingEnv(
            df=df_val,
            ticker="SPY",
            initial_amount=100_000,
            options_data_path=LOCAL_CACHE,
            tech_indicator_list=TECH_INDICATORS,
            episode_len=EPISODE_LEN,
        )

    venv = DummyVecEnv([make_train_env])
    venv_normed = VecNormalize(
        venv,
        norm_obs=True, norm_reward=False,
        clip_obs=10.0, training=True,
    )
    
    val_venv = DummyVecEnv([make_val_env])
    val_venv_normed = VecNormalize(
        val_venv,
        norm_obs=True, norm_reward=False,
        clip_obs=10.0, training=False,
    )
    # Sync normalization stats
    val_venv_normed.obs_rms = venv_normed.obs_rms
    
    # 4. Build Model
    model = SAC(
        "MlpPolicy",
        venv_normed,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        learning_starts=5_000,
        batch_size=BATCH_SIZE,
        tau=TAU,
        gamma=GAMMA,
        train_freq=1,
        gradient_steps=1,
        ent_coef=ENT_COEF,
        target_entropy="auto",
        policy_kwargs=dict(
            net_arch=dict(pi=NET_ARCH, qf=NET_ARCH)
        ),
        verbose=0,
        device="cuda",
    )

    num_params = sum(p.numel() for p in model.policy.parameters()) / 1e6
    
    # 5. Train Model
    t0_train = time.time()
    time_limit_cb = TimeLimitCallback(TIME_BUDGET)
    
    try:
        # We set total_timesteps arbitrarily high, but it will be cut off by TIME_BUDGET
        model.learn(total_timesteps=3_000_000, callback=time_limit_cb)
    except Exception as e:
        print(f"Training interrupted or failed: {e}")

    t1_train = time.time()
    train_seconds = t1_train - t0_train
    
    # 6. Evaluate Model
    val_venv_normed.obs_rms = venv_normed.obs_rms # Sync again just in case
    val_dsr, val_pnl = evaluate_policy(model, val_venv_normed)

    t1_total = time.time()
    total_seconds = t1_total - t0_total
    
    # Optional: peak vram
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        peak_vram_mb = 0.0

    # 7. Print summary exact format
    print("\n---")
    print(f"val_dsr:          {val_dsr:.6f}")
    print(f"val_pnl:          {val_pnl:.2f}")
    print(f"training_seconds: {train_seconds:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_steps:        {model.num_timesteps}")
    print(f"num_params_M:     {num_params:.2f}")
    print(f"episode_len:      {EPISODE_LEN}")


if __name__ == "__main__":
    main()

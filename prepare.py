"""
One-time data preparation and fixed evaluation for autoresearch SPY Options RL experiments.

Usage:
    python prepare.py
"""

import os
import sys
import time
import argparse
import pandas as pd
import numpy as np

from google.cloud import storage
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 10800  # training time budget in seconds (3 hours)
EVAL_EPISODES = 5  # Number of episodes for validation
TICKER = "SPY"
LOCAL_CACHE = "C:/tmp/options_chains/spy/"
GCS_PREFIX = "options_chains/spy/"

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def sync_data():
    """Download SPY options from GCS to local cache, and verify parquet."""
    load_dotenv()
    project_id = os.getenv("GCS_PROJECT_ID", "project-tft-777")
    bucket_name = os.getenv("GCS_BUCKET_NAME", "project_tft_pipeline")

    os.makedirs(LOCAL_CACHE, exist_ok=True)
    print(f"Data: Syncing SPY options from GCS (gs://{bucket_name}/{GCS_PREFIX})...")
    
    try:
        client = storage.Client(project=project_id)
        bucket = client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=GCS_PREFIX))
        downloaded = 0
        for b in blobs:
            fn = b.name.split("/")[-1]
            if not fn.endswith(".json"):
                continue
            lp = os.path.join(LOCAL_CACHE, fn)
            if not os.path.exists(lp):
                b.download_to_filename(lp)
                downloaded += 1
        print(f"Data: GCS sync complete. Downloaded {downloaded} new day files.")
    except Exception as e:
        print(f"Data: GCS sync failed: {e}. Using existing local cache.")

    # Check for parquet
    parquet_path = "C:/Users/cerod/.gemini/antigravity/scratch/riskpulse-backend/daily_features.parquet"
    if not os.path.exists(parquet_path):
        print(f"Data: WARNING: Parquet {parquet_path} not found.")
        # Try local repo directory
        parquet_path = "daily_features.parquet"
        if not os.path.exists(parquet_path):
            print(f"Data: WARNING: Parquet {parquet_path} also not found. Evaluation may fail.")
    
    return parquet_path

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def evaluate_policy(model, env, num_episodes=EVAL_EPISODES):
    """
    Evaluates the trained RL policy over a fixed number of episodes.
    Returns the average DSR (Differential Sharpe Ratio) over the validation episodes.
    """
    returns = []
    dsr_scores = []
    
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        
        while not done:
            # Model predicts action
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            # DummyVecEnv returns arrays for done and info
            if isinstance(done, np.ndarray):
                done = done[0]
            if isinstance(info, (list, tuple, np.ndarray)):
                info = info[0]
            
        if "episode" in info:
            returns.append(info["episode"]["r"])
        
        # Calculate DSR at end of episode based on asset memory
        asset_memory = env.get_attr('asset_memory')[0]
        if len(asset_memory) > 1:
            rets = np.diff(asset_memory) / np.array(asset_memory[:-1])
            if np.std(rets) > 0:
                sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252) # Annualized
                dsr_scores.append(sharpe)
            else:
                dsr_scores.append(0.0)
                
    avg_return = np.mean(returns) if returns else 0.0
    avg_dsr = np.mean(dsr_scores) if dsr_scores else 0.0
    
    return avg_dsr, avg_return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for autoresearch options RL")
    args = parser.parse_args()

    print("Step 1: Sync Data")
    sync_data()
    print("Done! Ready to train.")

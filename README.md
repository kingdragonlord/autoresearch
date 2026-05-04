# Autoresearch: SPY Options RL Framework

This repository is an autonomous AI agent experimentation framework designed to optimize a Stable-Baselines3 SAC reinforcement learning agent on a SPY Options Trading environment.

*This framework is heavily adapted from the original `autoresearch` LLM repo by @karpathy to be used for Option Trading.*

The idea: give an AI agent a training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. The goal is to maximize the risk-adjusted returns (Differential Sharpe Ratio).

## How it works

The repo is deliberately kept small and only really has four files that matter:

- **`prepare.py`** — data prep (syncs SPY options from GCS, validates parquet) and runtime evaluation metric (`evaluate_policy`). Not modified.
- **`env_optiontrading_spy.py`** — the SPY Options RL environment. Defines state space (195 dims), action space (per-bucket allocation), and step logic. Not modified.
- **`train.py`** — the single file the agent edits. Contains the SAC model, optimizer, hyperparameters, and training loop. Everything is fair game here. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for the agent. **This file is edited and iterated on by the human**.

Training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation). The metrics are **val_dsr** (Differential Sharpe Ratio) and **val_pnl** — higher is better.

## Quick start

```bash
# 1. Install dependencies (requires stable-baselines3, torch, pandas, gymnasium, etc)
uv pip install -r requirements.txt # or similar

# 2. Sync options data
python prepare.py

# 3. Manually run a single training experiment (~5 min)
python train.py
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

## Project structure

```
prepare.py                — data prep + evaluation utilities (do not modify)
env_optiontrading_spy.py  — RL environment (do not modify)
train.py                  — RL algorithm, hyperparameters, training loop (agent modifies this)
program.md                — agent instructions
pyproject.toml            — dependencies
```

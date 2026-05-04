# autoresearch

This is an experiment to have the LLM do its own research, adapted for **RL SPY Options Trading**.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, SPY options data prep, and evaluation metric. Do not modify.
   - `env_optiontrading_spy.py` - the SPY Options RL environment. You may inspect this to understand the state space and reward but mostly leave it alone unless requested.
   - `train.py` — the file you modify. RL algorithm (SAC), network architecture, hyperparameters, optimizer, training loop.
4. **Verify data exists**: Check that `C:/tmp/options_chains/spy/` contains data files. If not, tell the human to run `python prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `python train.py`.

**What you CAN do:**
- Modify `train.py` — this is the primary file you edit. Everything is fair game: SAC hyperparameters (gamma, tau, learning_rate, buffer_size, batch_size, ent_coef), network architecture (net_arch), custom callback modifications, modifying the reward scaling prior to feeding it to the agent, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation (`evaluate_policy`) and training constants (time budget).
- Install new packages or add dependencies.
- Modify the evaluation harness. The `evaluate_policy` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the highest val_dsr.** DSR stands for Differential Sharpe Ratio. Higher is better (indicating greater risk-adjusted returns). Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, hyperparameters, etc. The only constraint is that the code runs without crashing and finishes within the time budget.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_dsr:          1.523450
val_pnl:          1500.50
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     4506.2
num_steps:        95300
num_params_M:     1.2
episode_len:      21
```

Note that the script is configured to always stop after 5 minutes. You can extract the key metric from the log file:

```
grep "^val_dsr:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_dsr	val_pnl	status	description
```

1. git commit hash (short, 7 chars)
2. val_dsr achieved (e.g. 1.234567) — use -999.0 for crashes
3. val_pnl achieved (e.g. 150.2) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_dsr	val_pnl	status	description
a1b2c3d	1.234500	450.2	keep	baseline
b2c3d4e	1.450200	600.5	keep	decrease LR to 1e-5
c3d4e5f	1.300000	550.0	discard	increase buffer size
d4e5f6g	-999.000	0.0	crash	invalid net_arch shape
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `python train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_dsr:\|^val_pnl:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_dsr improved (higher), you "advance" the branch, keeping the git commit
9. If val_dsr is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate.

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?". The loop runs until the human interrupts you, period.

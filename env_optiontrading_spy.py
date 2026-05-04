"""
================================================================================
  env_optiontrading_spy.py  —  OptionTradingEnv (SPY Edition)
  SPY Options RL Environment  |  Cloned from env_optiontrading_v6.py
================================================================================
  Key SPY-specific tunings vs TSLA v6:
    - Ticker default: "SPY"
    - File naming:    spy_strikes_*.json  (matches scrape_spy_options.py output)
    - Strike OTM bands: ±5% / ±10%  (TSLA used ±10% / ±20% — SPY moves less)
    - Concentration cap: 20% per bucket  (vs 25% for TSLA)
    - SL threshold: 150% move against  (vs 120% — SPY needs more breathing room)
    - Regime thresholds: BULL >8%, BEAR <-5%  (vs >15% / <-10% for single stock)

  State space: 195 dims (identical to v6)
    cash(1) + spot(1) + drawdown(1) + tech(9) + alpha(3) + bucket×10×18(180)

  Reward: 0.5 × DSR + 0.5 × dollar_PnL (same as v6)
================================================================================
"""
import os
import json
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding


class OptionTradingEnv(gym.Env):
    """
    SPY Options RL Environment — adapted from env_optiontrading_v6.py.

    State: 195 dims (cash, spot, drawdown, 9 tech, 3 alpha, 10×18 bucket features)
    Action: [-1,1]^18 — per-bucket allocation fraction
    Reward: 50% Differential Sharpe Ratio + 50% dollar-PnL
    """
    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        df: pd.DataFrame,
        ticker: str = "SPY",
        initial_amount: int = 100_000,
        options_data_path: str = "C:/tmp/options_chains/spy/",
        reward_scaling: float = 1.0,
        tech_indicator_list: list = [],
        make_plots: bool = False,
        episode_len: int = 21,
    ):
        super(OptionTradingEnv, self).__init__()

        self.df                  = df
        self.date_array          = sorted(list(df.date.unique()))
        self.ticker              = ticker
        self.initial_amount      = initial_amount
        self.options_data_path   = options_data_path
        self.reward_scaling      = reward_scaling
        self.tech_indicator_list = tech_indicator_list
        self.make_plots          = make_plots
        self.episode_len         = episode_len

        self.num_buckets = 18

        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.num_buckets,), dtype=np.float32
        )

        # State: cash(1) + spot(1) + drawdown(1) + tech(9) + alpha(3) + 10×18(180) = 195
        self.state_space_dim = 1 + 1 + 1 + len(self.tech_indicator_list) + 3 + (self.num_buckets * 10)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space_dim,), dtype=np.float32
        )

        self.day      = 0
        self.terminal = False
        self._seed()

        # Pre-calculate all bucket matrices into RAM
        self.precalculated_buckets = {}
        self.precalculated_spots   = {}
        print(f"Precalculating option buckets for {len(self.date_array)} days...")
        for d in self.date_array:
            self.active_chain = self._load_data(d)
            b, s = self._sort_into_buckets()
            self.precalculated_buckets[d] = b
            self.precalculated_spots[d]   = s

        # Institutional Alpha Signals (IV-Rank, Surface Skew)
        print(f"Precalculating Alpha dimensions for {len(self.date_array)} days...")
        self.alpha_data = {}
        for d in self.date_array:
            day_buckets = self.precalculated_buckets[d]
            call_ivs = [b["iv"] for i, b in enumerate(day_buckets) if i % 6 < 3 and b["active"]]
            put_ivs  = [b["iv"] for i, b in enumerate(day_buckets) if i % 6 >= 3 and b["active"]]
            avg_call_iv = np.mean(call_ivs) if call_ivs else 0.5
            avg_put_iv  = np.mean(put_ivs)  if put_ivs  else 0.5
            self.alpha_data[d] = {
                "avg_iv": (avg_call_iv + avg_put_iv) / 2.0,
                "skew":    avg_call_iv - avg_put_iv,
            }

        # Rolling IV-Rank (30-day window)
        all_avg_ivs = [self.alpha_data[d]["avg_iv"] for d in self.date_array]
        for i, d in enumerate(self.date_array):
            window    = all_avg_ivs[max(0, i - 30): i + 1]
            min_iv, max_iv = min(window), max(window)
            iv_rank   = (all_avg_ivs[i] - min_iv) / (max_iv - min_iv + 1e-6)
            self.alpha_data[d]["iv_rank"] = iv_rank

        print("Computing regime classification...")
        self._compute_regime_map()

        print("SPY Environment Ready.")

    # ──────────────────────────────────────────────────────────────────────────
    def _compute_regime_map(self):
        """
        SPY-specific regime thresholds:
          BULL     : 63-day trailing return > 8%   (TSLA used 15% — index moves less)
          BEAR     : 63-day trailing return < -5%  (TSLA used -10%)
          VOLATILE : iv_rank > 0.70
          SIDEWAYS : everything else
        """
        price_by_date = {}
        for d in self.date_array:
            sub = self.df[self.df['date'] == d]
            if not sub.empty and 'close' in sub.columns:
                price_by_date[d] = float(sub.iloc[0]['close'])

        self._regime_map   = {}
        self._regime_dates = {'BULL': [], 'BEAR': [], 'VOLATILE': [], 'SIDEWAYS': []}

        for i, d in enumerate(self.date_array):
            if i < 63:
                regime = 'SIDEWAYS'
            else:
                past_63 = [price_by_date.get(self.date_array[j])
                           for j in range(i - 63, i) if self.date_array[j] in price_by_date]
                past_63 = [p for p in past_63 if p is not None]
                if len(past_63) < 20:
                    regime = 'SIDEWAYS'
                else:
                    trailing_return = (past_63[-1] - past_63[0]) / (past_63[0] + 1e-8)
                    iv_rank         = self.alpha_data[d]["iv_rank"]
                    # SPY-tuned thresholds (tighter than single-stock)
                    if trailing_return > 0.08:
                        regime = 'BULL'
                    elif trailing_return < -0.05:
                        regime = 'BEAR'
                    elif iv_rank > 0.70:
                        regime = 'VOLATILE'
                    else:
                        regime = 'SIDEWAYS'

            self._regime_map[d] = regime
            self._regime_dates[regime].append(d)

        for regime in self._regime_dates:
            if not self._regime_dates[regime]:
                self._regime_dates[regime] = list(self.date_array)

        regime_counts = {r: len(v) for r, v in self._regime_dates.items()}
        print(f"  Regime distribution: {regime_counts}")

    # ──────────────────────────────────────────────────────────────────────────
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # ──────────────────────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        regime     = np.random.choice(['BULL', 'BEAR', 'VOLATILE', 'SIDEWAYS'])
        regime_set = set(self._regime_dates[regime])

        valid_starts = [
            i for i, d in enumerate(self.date_array)
            if d in regime_set and i + self.episode_len < len(self.date_array) - 1
        ]
        if not valid_starts:
            valid_starts = list(range(max(1, len(self.date_array) - self.episode_len - 1)))

        self.day      = int(np.random.choice(valid_starts))
        self.ep_end   = min(self.day + self.episode_len, len(self.date_array) - 1)
        self.terminal = False

        self.cash            = self.initial_amount
        self.portfolio_value = self.cash
        self.max_net_worth   = self.initial_amount

        self.dsr_A = 0.0
        self.dsr_B = 0.0

        self.inventory      = {i: {"qty": 0, "avg_cost": 0.0, "symbol": ""} for i in range(self.num_buckets)}
        self.asset_memory   = [self.portfolio_value]
        self.rewards_memory = []
        self.date_memory    = [self.date_array[self.day]]

        self.state = self._build_state()
        return self.state, {}

    # ──────────────────────────────────────────────────────────────────────────
    def _load_data(self, date_str):
        """Load SPY options chain JSON from local cache."""
        filepath = os.path.join(
            self.options_data_path,
            f"spy_strikes_{date_str}.json"
        )
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                return data.get('data', data)
        except Exception:
            return []

    # ──────────────────────────────────────────────────────────────────────────
    def _sort_into_buckets(self):
        """
        Condense full options chain into 18 static vectors.

        SPY-specific strike bands (tighter than TSLA):
          ATM / OTM+5% / OTM+10%  for Calls
          ATM / OTM-5% / OTM-10%  for Puts
        """
        buckets = [
            {"active": 0.0, "price": 0.0, "strike": 0.0,
             "delta": 0.0, "theta": 0.0, "gamma": 0.0, "vega": 0.0,
             "iv": 0.0, "dte": 0.0, "symbol": ""}
            for _ in range(self.num_buckets)
        ]

        if not self.active_chain:
            return buckets, 0.0

        spot_price = float(self.active_chain[0].get("stockPrice", 0.0))
        if spot_price == 0.0:
            return buckets, 0.0

        df = pd.DataFrame(self.active_chain)
        if df.empty:
            return buckets, spot_price

        try:
            df_short = df[df['dte'] < 21]
            df_mid   = df[(df['dte'] >= 21) & (df['dte'] < 90)]
            df_long  = df[df['dte'] >= 90]

            def _map_bucket(sub_df, strike_target, is_call, b_idx):
                if sub_df.empty:
                    return
                volume_col = 'callVolume' if is_call else 'putVolume'
                sub = sub_df[sub_df[volume_col] > 0].copy()
                if sub.empty:
                    return
                sub['dist'] = abs(sub['strike'] - strike_target)
                closest = sub.loc[sub['dist'].idxmin()]
                buckets[b_idx].update({
                    "active": 1.0,
                    "price":  float(closest['callValue'] if is_call else closest['putValue']),
                    "strike": float(closest['strike']),
                    "delta":  float(closest.get('delta', 0)),
                    "theta":  float(closest.get('theta', 0)),
                    "gamma":  float(closest.get('gamma', 0)),
                    "vega":   float(closest.get('vega', 0)),
                    "iv":     float(closest.get('callMidIv' if is_call else 'putMidIv', 0)),
                    "dte":    float(closest.get('dte', 0)),
                })

            # SPY strike bands: ±5% / ±10% (tighter than TSLA's ±10% / ±20%)
            # Short expiry (Buckets 0-5)
            _map_bucket(df_short, spot_price,        True,  0)   # ATM Call
            _map_bucket(df_short, spot_price * 1.05, True,  1)   # +5% OTM Call
            _map_bucket(df_short, spot_price * 1.10, True,  2)   # +10% OTM Call
            _map_bucket(df_short, spot_price,        False, 3)   # ATM Put
            _map_bucket(df_short, spot_price * 0.95, False, 4)   # -5% OTM Put
            _map_bucket(df_short, spot_price * 0.90, False, 5)   # -10% OTM Put
            # Mid expiry (Buckets 6-11)
            _map_bucket(df_mid,   spot_price,        True,  6)
            _map_bucket(df_mid,   spot_price * 1.05, True,  7)
            _map_bucket(df_mid,   spot_price * 1.10, True,  8)
            _map_bucket(df_mid,   spot_price,        False, 9)
            _map_bucket(df_mid,   spot_price * 0.95, False, 10)
            _map_bucket(df_mid,   spot_price * 0.90, False, 11)
            # Long expiry (Buckets 12-17)
            _map_bucket(df_long,  spot_price,        True,  12)
            _map_bucket(df_long,  spot_price * 1.05, True,  13)
            _map_bucket(df_long,  spot_price * 1.10, True,  14)
            _map_bucket(df_long,  spot_price,        False, 15)
            _map_bucket(df_long,  spot_price * 0.95, False, 16)
            _map_bucket(df_long,  spot_price * 0.90, False, 17)

        except Exception:
            pass

        return buckets, spot_price

    # ──────────────────────────────────────────────────────────────────────────
    def _build_state(self):
        d = self.date_array[self.day]
        self.bucket_data = self.precalculated_buckets[d]
        spot_price       = self.precalculated_spots[d]

        drawdown_now = (self.max_net_worth - self.portfolio_value) / (self.max_net_worth + 1e-8)

        state = [
            float(self.cash)       / 1_000_000.0,
            float(spot_price)      / 1_000.0,
            float(np.clip(drawdown_now, 0.0, 1.0)),
        ]

        # Tech indicators — use exact same column names as TSLA v6
        # Verified in parquet: open, high, low, close, volume, macd_hist, atr_14, bb_width_20, roc_21
        sub_df = self.df[self.df['date'] == d]
        if not sub_df.empty:
            for tech in self.tech_indicator_list:
                val = float(sub_df.iloc[0][tech])
                if   tech == "volume":                              val /= 1e8
                elif tech in ("open", "high", "low", "close"):     val /= 1_000.0
                elif tech in ("macd_hist", "atr_14", "bb_width_20"): val /= 100.0
                state.append(val)
        else:
            state.extend([0.0] * len(self.tech_indicator_list))

        # Alpha surface
        alpha = self.alpha_data[d]
        state.extend([alpha["avg_iv"], alpha["iv_rank"], alpha["skew"]])

        # Per-bucket features: 10 dims × 18 = 180
        for i in range(self.num_buckets):
            b       = self.bucket_data[i]
            inv_qty = float(self.inventory[i]["qty"]) / 10.0

            strike = b["strike"]
            if i % 6 < 3:  # Calls
                intrinsic = max(0.0, spot_price - strike)
            else:           # Puts
                intrinsic = max(0.0, strike - spot_price)

            extrinsic       = max(0.0, b["price"] - intrinsic)
            extrinsic_ratio = extrinsic / (b["price"] + 1e-6)
            dte_norm        = float(np.clip(b["dte"] / 365.0, 0.0, 1.0))

            unrealized_pnl_pct = 0.0
            qty      = self.inventory[i]["qty"]
            avg_cost = self.inventory[i]["avg_cost"]
            if qty != 0 and avg_cost > 0 and b["active"]:
                current_val    = b["price"] * 100 * qty
                cost_basis_val = avg_cost * abs(qty)
                if qty > 0:
                    unrealized_pnl_pct = (current_val - cost_basis_val) / (cost_basis_val + 1e-8)
                else:
                    unrealized_pnl_pct = (cost_basis_val - abs(current_val)) / (cost_basis_val + 1e-8)
                unrealized_pnl_pct = float(np.clip(unrealized_pnl_pct, -2.0, 2.0))

            state.extend([
                b["active"],
                b["price"]  / 100.0,
                extrinsic_ratio,
                b["delta"],
                b["theta"]  / 10.0,
                b["gamma"]  * 10.0,
                b["vega"]   / 10.0,
                dte_norm,
                inv_qty,
                unrealized_pnl_pct,
            ])

        state_clipped = np.clip(np.array(state, dtype=np.float64), -3.4e38, 3.4e38)
        state = np.nan_to_num(state_clipped.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        return state

    # ──────────────────────────────────────────────────────────────────────────
    def step(self, actions):
        # Check terminal BEFORE incrementing — return early if already done
        if self.day >= self.ep_end:
            self.terminal = True
            return self.state, 0.0, True, False, {"terminal_observation": self.state}

        begin_total_asset = self.portfolio_value

        iv_rank_today = self.alpha_data[self.date_array[self.day]]["iv_rank"]
        regime_scale  = 1.0 - (0.6 * iv_rank_today)

        contracts_transacted = 0

        for i in range(self.num_buckets):
            act = float(actions[i])
            bk  = self.bucket_data[i]

            if bk["active"] == 0 or bk["price"] < 0.05:
                continue

            opt_price_dollars = bk["price"] * 100

            if act > 0:
                max_spend = self.portfolio_value * 0.10 * regime_scale
                max_buy_by_portfolio = max(0, int(max_spend * act / opt_price_dollars))
                max_buy_by_cash      = max(0, int(self.cash // opt_price_dollars))
                buy_qty = min(max_buy_by_portfolio, max_buy_by_cash)

                if buy_qty > 0:
                    existing_val  = max(0, self.inventory[i]["qty"]) * opt_price_dollars
                    # SPY concentration cap: 20% (vs 25% for TSLA)
                    cap_val       = self.portfolio_value * 0.20
                    remaining_cap = max(0.0, cap_val - existing_val)
                    max_buy_by_cap = int(remaining_cap / opt_price_dollars)
                    buy_qty = min(buy_qty, max_buy_by_cap)

                if buy_qty > 0:
                    cur_qty  = max(0, self.inventory[i]["qty"])
                    cur_cost = self.inventory[i]["avg_cost"]
                    new_total = (cur_qty * cur_cost) + (buy_qty * opt_price_dollars)
                    self.inventory[i]["avg_cost"] = new_total / (cur_qty + buy_qty)
                    self.cash                    -= buy_qty * opt_price_dollars
                    self.inventory[i]["qty"]     += buy_qty
                    contracts_transacted         += buy_qty

            elif act < 0:
                req_qty          = abs(int(self.portfolio_value * 0.10 * regime_scale
                                         * abs(act) / opt_price_dollars))
                req_qty          = max(1, req_qty)
                current_holdings = self.inventory[i]["qty"]

                if current_holdings > 0:
                    sell_qty = min(current_holdings, req_qty)
                    self.cash                += sell_qty * opt_price_dollars
                    self.inventory[i]["qty"] -= sell_qty
                    if self.inventory[i]["qty"] == 0:
                        self.inventory[i]["avg_cost"] = 0.0
                    contracts_transacted += sell_qty

                else:
                    triad_start = (i // 3) * 3
                    triad_idxs  = list(range(triad_start, triad_start + 3))
                    long_qty    = sum(self.inventory[idx]["qty"] for idx in triad_idxs
                                     if self.inventory[idx]["qty"] > 0)
                    short_qty   = sum(abs(self.inventory[idx]["qty"]) for idx in triad_idxs
                                     if self.inventory[idx]["qty"] < 0)
                    available   = max(0, long_qty - short_qty)
                    sto_qty     = min(available, req_qty)

                    if sto_qty > 0:
                        long_legs = [(idx, self.inventory[idx]["qty"])
                                     for idx in triad_idxs if self.inventory[idx]["qty"] > 0]
                        long_idx  = long_legs[0][0]
                        strike_w  = abs(self.bucket_data[i]["strike"] - self.bucket_data[long_idx]["strike"])
                        margin_req = strike_w * 100 * sto_qty

                        if self.cash >= margin_req:
                            cur_short = abs(self.inventory[i]["qty"])
                            cur_cost  = self.inventory[i]["avg_cost"]
                            new_total = (cur_short * cur_cost) + (sto_qty * opt_price_dollars)
                            self.inventory[i]["avg_cost"] = new_total / (cur_short + sto_qty)
                            self.cash                    += sto_qty * opt_price_dollars
                            self.inventory[i]["qty"]     -= sto_qty
                            contracts_transacted         += sto_qty

        today_bucket_data = self.bucket_data
        self.day += 1
        # Check terminal AFTER incrementing day
        self.terminal = self.day >= self.ep_end
        self.state = self._build_state()

        # Auto-closer
        auto_close_reward = 0.0
        for i in range(self.num_buckets):
            qty = self.inventory[i]["qty"]
            if qty == 0 or today_bucket_data[i]["active"] == 0.0:
                continue

            current_price = today_bucket_data[i]["price"] * 100
            avg_cost      = self.inventory[i]["avg_cost"]

            if qty < 0 and avg_cost > 0:
                if current_price <= 0.35 * avg_cost:        # 65% decay TP
                    close_qty           = abs(qty)
                    self.cash          -= close_qty * current_price
                    self.inventory[i]   = {"qty": 0, "avg_cost": 0.0, "symbol": ""}
                    auto_close_reward  += 2.0
                    contracts_transacted += close_qty

                # SPY SL: 150% move against (vs 120% for TSLA — index needs more room)
                elif current_price >= 2.50 * avg_cost:
                    close_qty           = abs(qty)
                    self.cash          -= close_qty * current_price
                    self.inventory[i]   = {"qty": 0, "avg_cost": 0.0, "symbol": ""}
                    auto_close_reward  -= 2.0
                    contracts_transacted += close_qty

            elif qty > 0 and avg_cost > 0:
                loss_pct = (current_price - avg_cost) / (avg_cost + 1e-8)
                if loss_pct <= -0.90:
                    close_qty           = qty
                    self.cash          += close_qty * current_price
                    self.inventory[i]   = {"qty": 0, "avg_cost": 0.0, "symbol": ""}
                    auto_close_reward  -= 1.0
                    contracts_transacted += close_qty

        # Mark-to-market
        end_total_asset = self.cash
        for i in range(self.num_buckets):
            qty = self.inventory[i]["qty"]
            if self.bucket_data[i]["active"] == 1.0:
                end_total_asset += qty * self.bucket_data[i]["price"] * 100

        self.portfolio_value = max(1.0, end_total_asset)
        self.max_net_worth   = max(self.max_net_worth, self.portfolio_value)

        if end_total_asset <= 0.0:
            self.terminal = True

        # DSR + dollar-PnL reward (identical to v6)
        eta        = 0.01
        log_return = np.log(self.portfolio_value / max(1.0, begin_total_asset))
        self.dsr_A = self.dsr_A + eta * (log_return    - self.dsr_A)
        self.dsr_B = self.dsr_B + eta * (log_return**2 - self.dsr_B)
        dsr_var    = max(0.0, self.dsr_B - self.dsr_A**2)
        dsr_step   = self.dsr_A / (np.sqrt(dsr_var) + 1e-8)
        dsr_reward = float(np.clip(dsr_step, -1.0, 1.0))

        pnl_reward       = (self.portfolio_value - begin_total_asset) / 1_000.0
        transaction_cost = contracts_transacted * 0.01
        cash_ratio       = self.cash / max(1.0, self.portfolio_value)
        cash_penalty     = cash_ratio * 0.002

        reward = (0.5 * dsr_reward
                  + 0.5 * pnl_reward
                  + auto_close_reward
                  - transaction_cost
                  - cash_penalty)

        reward = float(np.nan_to_num(reward, nan=-5.0))
        reward = float(np.clip(reward, -5.0, 5.0))

        if end_total_asset <= 0.0:
            reward -= 100.0

        self.rewards_memory.append(reward)
        self.asset_memory.append(self.portfolio_value)
        self.date_memory.append(self.date_array[self.day])

        info = {"portfolio_value": self.portfolio_value, "cash": self.cash}
        if self.terminal:
            # SB3 Monitor-compatible episode summary
            info["episode"] = {
                "r": float(sum(self.rewards_memory)),
                "l": len(self.rewards_memory),
                "p": float(self.portfolio_value),
            }
            info["terminal_observation"] = self.state

        return self.state, reward, self.terminal, False, info

    # ──────────────────────────────────────────────────────────────────────────
    def render(self, mode="human"):
        pass

    def close(self):
        pass

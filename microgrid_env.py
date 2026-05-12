"""
microgrid_env.py
================
Custom OpenAI Gym environment simulating a campus solar microgrid.

Components:
  - Solar PV panel (max 1000W)
  - LiFePO4 Battery (5 kWh capacity, 500W max charge/discharge)
  - Variable campus load (100–800W)
  - Grid connection with time-of-use pricing

State  : [pv_norm, load_norm, battery_soc, hour_sin, hour_cos, price_norm]
Actions: 0=Charge battery | 1=Discharge battery | 2=Solar-first (idle)
Reward : Minimize grid cost + maximize solar usage + protect battery health
"""

import numpy as np
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces


class MicrogridEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    # ------------------------------------------------------------------ #
    #  System parameters                                                   #
    # ------------------------------------------------------------------ #
    BATTERY_CAPACITY_WH   = 5000   # Wh
    BATTERY_MAX_POWER_W   = 500    # W  (max charge / discharge rate)
    BATTERY_EFFICIENCY    = 0.95
    SOC_MIN               = 0.20   # 20 % lower bound
    SOC_MAX               = 0.95   # 95 % upper bound

    PV_MAX_W              = 700    # W  (more realistic campus panel)
    LOAD_MIN_W            = 150    # W
    LOAD_MAX_W            = 900    # W  (higher evening loads)

    GRID_PEAK_PRICE       = 0.30   # $/kWh  evening peak 17:00–22:00
    GRID_OFFPEAK_PRICE    = 0.08   # $/kWh  cheap overnight and midday
    SOLAR_VALUE           = 0.12   # $/kWh  (avoided-cost reward)

    TIMESTEP_H            = 1.0    # 1 hour per step

    # ------------------------------------------------------------------ #
    def __init__(self, episode_length: int = 24 * 7, seed: int = 42):
        """
        Parameters
        ----------
        episode_length : int   Steps per episode (default = 1 week = 168 h)
        seed           : int   Random seed for reproducibility
        """
        super().__init__()
        self.episode_length = episode_length
        self.rng = np.random.default_rng(seed)

        # --- Action space: 4 discrete decisions ---
        # 0 = CHARGE      – solar surplus → battery; grid covers load deficit only
        # 1 = DISCHARGE   – battery covers load; grid as last resort
        # 2 = IDLE        – solar covers load; grid covers deficit; battery unchanged
        # 3 = GRID_CHARGE – actively import cheap grid power to fill battery
        self.action_space = spaces.Discrete(4)
        self.ACTION_LABELS = ["CHARGE", "DISCHARGE", "IDLE", "GRID_CHARGE"]

        # --- Observation space ---
        # [pv_norm, load_norm, soc, hour_sin, hour_cos, price_norm]
        low  = np.array([0, 0, 0, -1, -1, 0], dtype=np.float32)
        high = np.array([1, 1, 1,  1,  1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Internal state (populated by reset / step)
        self._battery_soc    = 0.5
        self._current_hour   = 0
        self._current_step   = 0
        self._current_pv     = 0.0
        self._current_load   = 0.0
        self._current_price  = self.GRID_OFFPEAK_PRICE
        self.history         = []

        # Episode-level accumulators
        self.ep_grid_cost    = 0.0
        self.ep_solar_used   = 0.0
        self.ep_solar_gen    = 0.0

    # ------------------------------------------------------------------ #
    #  Data generators                                                     #
    # ------------------------------------------------------------------ #
    def _solar_power(self, hour: float) -> float:
        """Gaussian bell-curve centred at solar noon (12:00), with noise."""
        if 6 <= hour <= 18:
            base  = self.PV_MAX_W * np.exp(-0.5 * ((hour - 12) / 3.0) ** 2)
            noise = self.rng.normal(0, base * 0.10)
            return float(np.clip(base + noise, 0, self.PV_MAX_W))
        return 0.0

    def _campus_load(self, hour: float, day_of_week: int) -> float:
        """Typical campus load: morning + lunch + evening peaks."""
        base    = 280.0
        morning = 380.0 * np.exp(-0.5 * ((hour -  9) / 1.5) ** 2)
        lunch   = 180.0 * np.exp(-0.5 * ((hour - 13) / 1.0) ** 2)
        evening = 340.0 * np.exp(-0.5 * ((hour - 19) / 2.0) ** 2)
        load    = base + morning + lunch + evening
        if day_of_week >= 5:          # Weekend — lower campus activity
            load *= 0.55
        noise = self.rng.normal(0, 20)
        return float(np.clip(load + noise, self.LOAD_MIN_W, self.LOAD_MAX_W))

    def _grid_price(self, hour: float) -> float:
        # Evening peak (high demand, no solar): 17:00-22:00
        return self.GRID_PEAK_PRICE if 17 <= hour <= 22 else self.GRID_OFFPEAK_PRICE

    # ------------------------------------------------------------------ #
    #  State builder                                                       #
    # ------------------------------------------------------------------ #
    def _build_obs(self) -> np.ndarray:
        hour = self._current_hour % 24
        day  = (self._current_step  // 24) % 7

        pv    = self._solar_power(hour)
        load  = self._campus_load(hour, day)
        price = self._grid_price(hour)

        # Cache for use inside step()
        self._current_pv    = pv
        self._current_load  = load
        self._current_price = price

        h_sin = float(np.sin(2 * np.pi * hour / 24))
        h_cos = float(np.cos(2 * np.pi * hour / 24))
        p_norm = (price - self.GRID_OFFPEAK_PRICE) / (
            self.GRID_PEAK_PRICE - self.GRID_OFFPEAK_PRICE
        )

        return np.array(
            [
                pv    / self.PV_MAX_W,
                load  / self.LOAD_MAX_W,
                self._battery_soc,
                h_sin,
                h_cos,
                p_norm,
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------ #
    #  Core Gym methods                                                    #
    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._battery_soc   = self.rng.uniform(0.40, 0.60)
        self._current_step  = 0
        self._current_hour  = int(self.rng.integers(0, 24))
        self.ep_grid_cost   = 0.0
        self.ep_solar_used  = 0.0
        self.ep_solar_gen   = 0.0
        self.history        = []
        return self._build_obs(), {}

    def step(self, action: int):
        pv    = self._current_pv
        load  = self._current_load
        price = self._current_price
        dt    = self.TIMESTEP_H

        # -------- Energy dispatch logic -------------------------------- #
        if action == 0:   # CHARGE
            solar_to_load    = min(pv, load)
            max_chg          = (
                (self.SOC_MAX - self._battery_soc)
                * self.BATTERY_CAPACITY_WH / dt
            )
            solar_to_bat     = min(
                max(0.0, pv - solar_to_load),
                self.BATTERY_MAX_POWER_W,
                max(0.0, max_chg),
            )
            grid_import      = max(0.0, load - solar_to_load)
            bat_delta_wh     = solar_to_bat * self.BATTERY_EFFICIENCY * dt
            solar_used       = solar_to_load + solar_to_bat

        elif action == 1: # DISCHARGE
            solar_to_load    = min(pv, load)
            deficit          = load - solar_to_load
            avail_wh         = max(
                0.0,
                (self._battery_soc - self.SOC_MIN) * self.BATTERY_CAPACITY_WH,
            )
            bat_power        = min(deficit, self.BATTERY_MAX_POWER_W,
                                   avail_wh / dt)
            grid_import      = max(0.0, deficit - bat_power)
            bat_delta_wh     = -(bat_power / self.BATTERY_EFFICIENCY) * dt
            solar_used       = solar_to_load

        elif action == 2: # IDLE
            solar_to_load    = min(pv, load)
            grid_import      = max(0.0, load - solar_to_load)
            bat_delta_wh     = 0.0
            solar_used       = solar_to_load

        else:             # action == 3: GRID_CHARGE (price arbitrage)
            # Actively import cheap grid power to fill battery
            solar_to_load    = min(pv, load)
            grid_for_load    = max(0.0, load - solar_to_load)
            max_chg          = (
                (self.SOC_MAX - self._battery_soc)
                * self.BATTERY_CAPACITY_WH / dt
            )
            grid_to_bat      = min(self.BATTERY_MAX_POWER_W, max(0.0, max_chg))
            grid_import      = grid_for_load + grid_to_bat
            bat_delta_wh     = grid_to_bat * self.BATTERY_EFFICIENCY * dt
            solar_used       = solar_to_load

        # -------- Battery SOC update ----------------------------------- #
        self._battery_soc = np.clip(
            self._battery_soc + bat_delta_wh / self.BATTERY_CAPACITY_WH,
            0.0, 1.0,
        )

        # -------- Reward ----------------------------------------------- #
        grid_cost = grid_import * price * self.TIMESTEP_H / 1000.0   # $

        # Penalise grid cost: peak-hour imports cost extra
        price_multiplier   = 4.0 if price >= self.GRID_PEAK_PRICE else 1.0
        grid_cost_penalty  = grid_import * price * self.TIMESTEP_H / 1000.0 * price_multiplier

        # Solar bonus
        solar_bonus = (solar_used / max(pv, 1.0)) * 0.03

        # Battery protection
        deep_penalty = 1.5 if self._battery_soc < self.SOC_MIN else 0.0
        over_penalty = 0.5 if self._battery_soc > self.SOC_MAX else 0.0

        # Arbitrage penalty: punish GRID_CHARGE during expensive peak hours
        arb_penalty = 2.0 if (action == 3 and price >= self.GRID_PEAK_PRICE) else 0.0

        reward = -grid_cost_penalty + solar_bonus - deep_penalty - over_penalty - arb_penalty

        grid_cost = grid_import * price * self.TIMESTEP_H / 1000.0
        # -------- Accumulate episode stats ----------------------------- #
        self.ep_grid_cost  += grid_cost
        self.ep_solar_used += solar_used
        self.ep_solar_gen  += pv

        # -------- Log -------------------------------------------------- #
        self.history.append(
            {
                "step":         self._current_step,
                "hour":         self._current_hour % 24,
                "pv_w":         pv,
                "load_w":       load,
                "soc":          self._battery_soc,
                "grid_w":       grid_import,
                "solar_used_w": solar_used,
                "bat_delta_wh": bat_delta_wh,
                "action":       action,
                "price":        price,
                "grid_cost":    grid_cost,
                "reward":       reward,
            }
        )

        # -------- Advance time ----------------------------------------- #
        self._current_step += 1
        self._current_hour  = (self._current_hour + 1) % 24
        done = self._current_step >= self.episode_length

        obs  = self._build_obs() if not done else np.zeros(6, dtype=np.float32)
        info = {
            "grid_cost":   grid_cost,
            "solar_used":  solar_used,
            "pv_generated": pv,
            "load":        load,
            "soc":         self._battery_soc,
        }
        return obs, reward, done, False, info

    def render(self, mode="human"):
        if not self.history:
            return
        r = self.history[-1]
        print(
            f"Step {r['step']:3d} | H:{r['hour']:02d}:00 | "
            f"PV:{r['pv_w']:6.1f}W  Load:{r['load_w']:6.1f}W | "
            f"SOC:{r['soc']*100:5.1f}%  Grid:{r['grid_w']:6.1f}W | "
            f"Act:{self.ACTION_LABELS[r['action']]:10s} | "
            f"R:{r['reward']:+.4f}"
        )

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #
    def solar_self_consumption_ratio(self) -> float:
        """Fraction of generated solar energy actually consumed on-site."""
        if self.ep_solar_gen < 1e-6:
            return 0.0
        return self.ep_solar_used / self.ep_solar_gen

    def summary(self) -> dict:
        return {
            "total_grid_cost_$":        round(self.ep_grid_cost, 4),
            "solar_self_consumption_%": round(self.solar_self_consumption_ratio() * 100, 2),
            "total_solar_generated_wh": round(self.ep_solar_gen, 1),
            "total_solar_used_wh":      round(self.ep_solar_used, 1),
        }

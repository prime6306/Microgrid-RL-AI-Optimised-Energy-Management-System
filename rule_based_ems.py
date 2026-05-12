"""
rule_based_ems.py
=================
Threshold-based (rule-based) Energy Management System.
Used as the baseline to compare against the RL agent.

Rules:
  1. If solar > load AND battery SOC < 0.90  → CHARGE
  2. If solar < load AND battery SOC > 0.30  → DISCHARGE
  3. Otherwise                               → IDLE (solar first, grid fills gap)

This mirrors the classical "if-then" EMS found in most deployed systems.
"""

import numpy as np
from microgrid_env import MicrogridEnv


class RuleBasedEMS:
    """
    Deterministic rule-based energy management controller.
    No learning — just fixed thresholds.
    """

    def __init__(
        self,
        charge_threshold: float = 0.90,     # SOC above which we stop charging
        discharge_threshold: float = 0.30,  # SOC below which we stop discharging
        excess_threshold_w: float = 50.0,   # W  surplus before we bother charging
    ):
        self.charge_threshold     = charge_threshold
        self.discharge_threshold  = discharge_threshold
        self.excess_threshold_w   = excess_threshold_w

    def select_action(self, obs: np.ndarray) -> int:
        """
        Map the current observation to one of the 3 actions.

        obs layout  (from MicrogridEnv):
          [0] pv_norm      (0–1)
          [1] load_norm    (0–1)
          [2] battery_soc  (0–1)
          [3] hour_sin
          [4] hour_cos
          [5] price_norm   (0–1)
        """
        pv_norm   = obs[0]
        load_norm = obs[1]
        soc       = obs[2]
        price_n   = obs[5]

        pv_w   = pv_norm   * MicrogridEnv.PV_MAX_W
        load_w = load_norm * MicrogridEnv.LOAD_MAX_W
        excess = pv_w - load_w

        # Rule 1: Surplus solar + battery not full → CHARGE
        if excess > self.excess_threshold_w and soc < self.charge_threshold:
            return 0  # CHARGE

        # Rule 2: Solar deficit + battery has charge → DISCHARGE
        if excess < 0 and soc > self.discharge_threshold:
            return 1  # DISCHARGE

        # Rule 3: Default — solar covers load, grid for deficit
        return 2  # IDLE


# ------------------------------------------------------------------ #
#  Standalone evaluation helper                                        #
# ------------------------------------------------------------------ #
def evaluate_rule_based(
    n_episodes: int = 10,
    episode_length: int = 24 * 7,
    seed: int = 0,
    verbose: bool = False,
) -> dict:
    """
    Run the rule-based EMS for n_episodes and collect metrics.

    Returns
    -------
    dict with keys:
        mean_grid_cost, mean_solar_ssc, all_grid_costs, all_solar_sscs
    """
    controller = RuleBasedEMS()
    grid_costs  = []
    solar_sscs  = []

    for ep in range(n_episodes):
        env  = MicrogridEnv(episode_length=episode_length, seed=seed + ep)
        obs, _  = env.reset()
        done = False

        while not done:
            action = controller.select_action(obs)
            obs, _, done, _, _ = env.step(action)
            if verbose:
                env.render()

        metrics = env.summary()
        grid_costs.append(metrics["total_grid_cost_$"])
        solar_sscs.append(metrics["solar_self_consumption_%"])

        if verbose or ep == 0:
            print(f"[Rule-Based] Episode {ep+1:2d} | "
                  f"Grid cost: ${metrics['total_grid_cost_$']:.3f} | "
                  f"Solar SSC: {metrics['solar_self_consumption_%']:.1f}%")

    return {
        "mean_grid_cost":  float(np.mean(grid_costs)),
        "mean_solar_ssc":  float(np.mean(solar_sscs)),
        "all_grid_costs":  grid_costs,
        "all_solar_sscs":  solar_sscs,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  Rule-Based EMS Baseline Evaluation")
    print("=" * 60)
    results = evaluate_rule_based(n_episodes=5, verbose=False)
    print(f"\nMean Grid Cost : ${results['mean_grid_cost']:.4f}")
    print(f"Mean Solar SSC : {results['mean_solar_ssc']:.2f}%")

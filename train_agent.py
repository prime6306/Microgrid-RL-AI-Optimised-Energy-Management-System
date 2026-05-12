"""
train_agent.py
==============
Train a DQN (and optionally PPO) RL agent on the MicrogridEnv
using Stable-Baselines3.

Usage
-----
    python train_agent.py                    # default: DQN, 100k steps
    python train_agent.py --algo ppo         # use PPO instead
    python train_agent.py --steps 200000     # longer training
    python train_agent.py --eval             # run evaluation after training
"""

import argparse
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from microgrid_env import MicrogridEnv

# ------------------------------------------------------------------ #
#  Custom training callback – logs reward every N steps              #
# ------------------------------------------------------------------ #
class TrainingLogger(BaseCallback):
    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq     = log_freq
        self.rewards      = []
        self.timesteps    = []
        self._ep_rewards  = []

    def _on_step(self) -> bool:
        # Collect episode rewards from infos
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._ep_rewards.append(info["episode"]["r"])

        if self.num_timesteps % self.log_freq == 0 and self._ep_rewards:
            mean_r = np.mean(self._ep_rewards[-10:])
            self.rewards.append(mean_r)
            self.timesteps.append(self.num_timesteps)
            if self.verbose:
                print(f"  Timestep {self.num_timesteps:7d} | "
                      f"Mean ep reward (last 10): {mean_r:+.4f}")
        return True


# ------------------------------------------------------------------ #
#  Network architecture helper                                        #
# ------------------------------------------------------------------ #
def make_policy_kwargs(algo: str) -> dict:
    """
    Small net suitable for Pi deployment.
    Two hidden layers of 128 units each.
    """
    return dict(net_arch=[128, 128])


# ------------------------------------------------------------------ #
#  Training                                                           #
# ------------------------------------------------------------------ #
def train(
    algo: str          = "dqn",
    total_steps: int   = 100_000,
    episode_length: int = 24 * 7,
    seed: int          = 42,
    save_dir: str      = "models",
    verbose: int       = 1,
) -> tuple:
    """
    Train an RL agent on MicrogridEnv.

    Returns (model, callback)
    """
    os.makedirs(save_dir, exist_ok=True)

    # --- Wrap environment ---
    def make_env():
        env = MicrogridEnv(episode_length=24, seed=seed)   # Daily episodes for faster learning
        env = Monitor(env)
        return env

    vec_env = DummyVecEnv([make_env])

    # --- Build model ---
    algo = algo.lower()
    common_kwargs = dict(
        policy       = "MlpPolicy",
        env          = vec_env,
        seed         = seed,
        verbose      = 0,
        policy_kwargs= make_policy_kwargs(algo),
    )

    if algo == "dqn":
        model = DQN(
            **common_kwargs,
            learning_rate       = 1e-3,
            buffer_size         = 50_000,
            learning_starts     = 1_000,
            batch_size          = 64,
            tau                 = 1.0,
            gamma               = 0.99,
            train_freq          = 4,
            target_update_interval = 500,
            exploration_fraction   = 0.20,
            exploration_final_eps  = 0.05,
        )
    elif algo == "ppo":
        model = PPO(
            **common_kwargs,
            learning_rate  = 3e-4,
            n_steps        = 2048,
            batch_size     = 64,
            n_epochs       = 10,
            gamma          = 0.99,
            gae_lambda     = 0.95,
            clip_range     = 0.2,
            ent_coef       = 0.01,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}. Choose 'dqn' or 'ppo'.")

    # --- Callback ---
    callback = TrainingLogger(log_freq=2000, verbose=verbose)

    # --- Train ---
    print(f"\n{'='*60}")
    print(f"  Training {algo.upper()} agent — {total_steps:,} steps")
    print(f"{'='*60}")
    t0 = time.time()
    model.learn(total_timesteps=total_steps, callback=callback, progress_bar=True)
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s")

    # --- Save ---
    model_path = os.path.join(save_dir, f"{algo}_microgrid")
    model.save(model_path)
    print(f"Model saved → {model_path}.zip")

    # --- Plot training curve ---
    if callback.timesteps:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(callback.timesteps, callback.rewards, color="#2ecc71", linewidth=2)
        ax.set_xlabel("Training Timesteps")
        ax.set_ylabel("Mean Episode Reward (last 10 eps)")
        ax.set_title(f"{algo.upper()} Training Curve — Microgrid EMS")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        curve_path = os.path.join(save_dir, f"{algo}_training_curve.png")
        fig.savefig(curve_path, dpi=150)
        plt.close(fig)
        print(f"Training curve → {curve_path}")

    return model, callback


# ------------------------------------------------------------------ #
#  Quick evaluation after training                                    #
# ------------------------------------------------------------------ #
def evaluate_agent(
    model,
    algo: str           = "dqn",
    n_episodes: int     = 10,
    episode_length: int = 24 * 7,
    seed: int           = 99,
) -> dict:
    grid_costs = []
    solar_sscs = []
    total_rewards = []

    for ep in range(n_episodes):
        env  = MicrogridEnv(episode_length=episode_length, seed=seed + ep)
        obs, _  = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(int(action))
            ep_reward += reward

        metrics = env.summary()
        grid_costs.append(metrics["total_grid_cost_$"])
        solar_sscs.append(metrics["solar_self_consumption_%"])
        total_rewards.append(ep_reward)

        print(f"[{algo.upper()}] Episode {ep+1:2d} | "
              f"Grid cost: ${metrics['total_grid_cost_$']:.3f} | "
              f"Solar SSC: {metrics['solar_self_consumption_%']:.1f}% | "
              f"Reward: {ep_reward:+.3f}")

    return {
        "mean_grid_cost":   float(np.mean(grid_costs)),
        "mean_solar_ssc":   float(np.mean(solar_sscs)),
        "mean_reward":      float(np.mean(total_rewards)),
        "all_grid_costs":   grid_costs,
        "all_solar_sscs":   solar_sscs,
    }


# ------------------------------------------------------------------ #
#  CLI entry point                                                     #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL agent for Microgrid EMS")
    parser.add_argument("--algo",  default="dqn",    choices=["dqn", "ppo"])
    parser.add_argument("--steps", default=100_000,  type=int)
    parser.add_argument("--seed",  default=42,       type=int)
    parser.add_argument("--eval",  action="store_true")
    args = parser.parse_args()

    model, _ = train(
        algo         = args.algo,
        total_steps  = args.steps,
        seed         = args.seed,
    )

    if args.eval:
        print(f"\n{'='*60}")
        print(f"  Evaluating trained {args.algo.upper()} agent")
        print(f"{'='*60}")
        results = evaluate_agent(model, algo=args.algo, n_episodes=5)
        print(f"\nMean Grid Cost : ${results['mean_grid_cost']:.4f}")
        print(f"Mean Solar SSC : {results['mean_solar_ssc']:.2f}%")
        print(f"Mean Reward    : {results['mean_reward']:.4f}")

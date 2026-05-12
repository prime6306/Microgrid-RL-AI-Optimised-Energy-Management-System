"""
main.py
=======
Master pipeline: trains the RL agent then runs full comparison evaluation.

Usage
-----
    python main.py                         # DQN, 100k steps, 10 eval episodes
    python main.py --algo ppo --steps 150000
    python main.py --quick                 # fast smoke-test (10k steps, 3 eps)
"""

import argparse
import os
import sys

import numpy as np

from microgrid_env  import MicrogridEnv
from rule_based_ems import RuleBasedEMS, evaluate_rule_based
from train_agent    import train, evaluate_agent
from evaluate       import (
    collect_metrics, run_episode,
    plot_comparison, plot_episode_timeline, plot_action_distribution,
    print_summary_table,
)


def quick_sanity_check():
    """Verify the environment works before doing anything heavy."""
    print("\n[Sanity check] Running 5 steps of MicrogridEnv …")
    env = MicrogridEnv(episode_length=5, seed=0)
    obs = env.reset()
    assert obs.shape == (6,), f"Unexpected obs shape: {obs.shape}"
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
    print("[Sanity check] PASSED ✓\n")


def main():
    parser = argparse.ArgumentParser(description="Microgrid RL — Full Pipeline")
    parser.add_argument("--algo",     default="dqn", choices=["dqn","ppo"])
    parser.add_argument("--steps",    default=100_000, type=int,
                        help="RL training timesteps")
    parser.add_argument("--episodes", default=10, type=int,
                        help="Evaluation episodes")
    parser.add_argument("--quick",    action="store_true",
                        help="Smoke-test with 10 k steps + 3 episodes")
    parser.add_argument("--seed",     default=42, type=int)
    args = parser.parse_args()

    if args.quick:
        args.steps    = 10_000
        args.episodes = 3

    # 0. Sanity
    quick_sanity_check()

    # 1. Rule-based baseline
    print("=" * 60)
    print("  STEP 1 — Rule-Based Baseline")
    print("=" * 60)
    rb_results = evaluate_rule_based(
        n_episodes=args.episodes, verbose=False
    )
    print(f"Mean Grid Cost : ${rb_results['mean_grid_cost']:.4f}")
    print(f"Mean Solar SSC : {rb_results['mean_solar_ssc']:.2f}%")

    # 2. Train RL agent
    print("\n" + "=" * 60)
    print(f"  STEP 2 — Training {args.algo.upper()} Agent")
    print("=" * 60)
    model, cb = train(
        algo         = args.algo,
        total_steps  = args.steps,
        seed         = args.seed,
        verbose      = 1,
    )

    # 3. Evaluate RL agent
    print("\n" + "=" * 60)
    print(f"  STEP 3 — Evaluating {args.algo.upper()} Agent")
    print("=" * 60)
    rl_results = evaluate_agent(
        model,
        algo         = args.algo,
        n_episodes   = args.episodes,
        seed         = args.seed + 100,
    )

    # 4. Gather metrics for plots
    rb_ctrl    = RuleBasedEMS()
    rb_metrics = collect_metrics(
        rb_ctrl, n_episodes=args.episodes, seed_start=300, is_rl=False
    )
    rl_metrics = collect_metrics(
        model, n_episodes=args.episodes, seed_start=300, is_rl=True
    )

    # Attach mean_reward (not computed inside collect_metrics — add it)
    rb_metrics["mean_reward"] = rb_results["mean_grid_cost"] * -1   # proxy
    rl_metrics["mean_reward"] = rl_results["mean_reward"]

    # 5. Generate all plots
    print("\n" + "=" * 60)
    print("  STEP 4 — Generating Plots")
    print("=" * 60)
    rb_df, _ = run_episode(rb_ctrl, seed=42, is_rl=False)
    rl_df, _ = run_episode(model,   seed=42, is_rl=True)

    os.makedirs("results", exist_ok=True)
    plot_comparison(rb_metrics, rl_metrics, args.algo)
    plot_episode_timeline(rb_df, rl_df, args.algo)
    plot_action_distribution(rb_df, rl_df, args.algo)

    # 6. Final summary
    print_summary_table(rb_metrics, rl_metrics, args.algo)

    print(f"""
╔══════════════════════════════════════════════════╗
║           Pipeline Complete ✓                   ║
╠══════════════════════════════════════════════════╣
║  Trained model  →  models/{args.algo}_microgrid.zip  ║
║  Plots          →  results/                       ║
╚══════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()

"""
evaluate.py
===========
Compare Rule-Based EMS vs. Trained RL Agent side-by-side.

Produces:
  results/comparison_bar.png       – bar chart: grid cost & solar SSC
  results/episode_timeline.png     – hourly energy flows for one episode
  results/action_distribution.png  – pie/bar of action frequencies
  results/soc_trajectory.png       – battery SOC over an episode

Usage
-----
    python evaluate.py --algo dqn
    python evaluate.py --algo ppo --episodes 10
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

from microgrid_env    import MicrogridEnv
from rule_based_ems   import RuleBasedEMS

# Try to load stable_baselines3; gracefully degrade if not installed
try:
    from stable_baselines3 import DQN, PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("[WARN] stable_baselines3 not found — RL agent evaluation skipped.")

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

ACTION_LABELS = ["CHARGE", "DISCHARGE", "IDLE", "GRID_CHARGE"]
ACTION_COLORS = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]


# ------------------------------------------------------------------ #
#  Run one full episode and return history DataFrame                  #
# ------------------------------------------------------------------ #
def run_episode(controller, episode_length=24*7, seed=7, is_rl=False):
    env  = MicrogridEnv(episode_length=episode_length, seed=seed)
    obs, _  = env.reset()
    done = False

    while not done:
        if is_rl:
            action, _ = controller.predict(obs, deterministic=True)
            action = int(action)
        else:
            action = controller.select_action(obs)
        obs, _, done, _, _ = env.step(action)

    df = pd.DataFrame(env.history)
    summary = env.summary()
    return df, summary


# ------------------------------------------------------------------ #
#  Multi-episode metric collection                                    #
# ------------------------------------------------------------------ #
def collect_metrics(controller, n_episodes=10, episode_length=24*7,
                    seed_start=0, is_rl=False):
    grid_costs, solar_sscs, rewards_list = [], [], []

    for ep in range(n_episodes):
        df, summary = run_episode(
            controller, episode_length=episode_length,
            seed=seed_start + ep, is_rl=is_rl
        )
        grid_costs.append(summary["total_grid_cost_$"])
        solar_sscs.append(summary["solar_self_consumption_%"])
        rewards_list.append(df["reward"].sum())

    return {
        "mean_grid_cost":  np.mean(grid_costs),
        "std_grid_cost":   np.std(grid_costs),
        "mean_solar_ssc":  np.mean(solar_sscs),
        "std_solar_ssc":   np.std(solar_sscs),
        "mean_reward":     np.mean(rewards_list),
        "all_grid_costs":  grid_costs,
        "all_solar_sscs":  solar_sscs,
    }


# ------------------------------------------------------------------ #
#  Plot 1 – Bar comparison                                            #
# ------------------------------------------------------------------ #
def plot_comparison(rb_metrics, rl_metrics, algo_name, save=True):
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle(
        f"Rule-Based EMS  vs.  {algo_name.upper()} Agent\n"
        f"(Microgrid EMS — {len(rb_metrics['all_grid_costs'])} episodes each)",
        fontsize=13, fontweight="bold"
    )

    labels = ["Rule-Based", algo_name.upper()]
    colors = ["#e67e22", "#27ae60"]

    # Grid cost
    ax = axes[0]
    vals  = [rb_metrics["mean_grid_cost"], rl_metrics["mean_grid_cost"]]
    errs  = [rb_metrics["std_grid_cost"],  rl_metrics["std_grid_cost"]]
    bars  = ax.bar(labels, vals, color=colors, width=0.4, yerr=errs, capsize=5)
    ax.set_ylabel("Avg Grid Cost ($)")
    ax.set_title("Grid Energy Cost ↓")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.001,
                f"${v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(vals)*1.3)
    ax.grid(axis="y", alpha=0.3)

    # Solar SSC
    ax = axes[1]
    vals  = [rb_metrics["mean_solar_ssc"], rl_metrics["mean_solar_ssc"]]
    errs  = [rb_metrics["std_solar_ssc"],  rl_metrics["std_solar_ssc"]]
    bars  = ax.bar(labels, vals, color=colors, width=0.4, yerr=errs, capsize=5)
    ax.set_ylabel("Solar Self-Consumption (%)")
    ax.set_title("Solar Utilisation ↑")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.3,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    # Mean episode reward
    ax = axes[2]
    vals  = [rb_metrics["mean_reward"], rl_metrics["mean_reward"]]
    bars  = ax.bar(labels, vals, color=colors, width=0.4)
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Cumulative Reward ↑")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + abs(v)*0.02,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "comparison_bar.png")
    if save:
        fig.savefig(path, dpi=150)
        print(f"Saved → {path}")
    plt.close(fig)


# ------------------------------------------------------------------ #
#  Plot 2 – Episode timeline (energy flows)                           #
# ------------------------------------------------------------------ #
def plot_episode_timeline(rb_df, rl_df, algo_name, save=True):
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.30)

    titles_row = ["Rule-Based EMS", f"{algo_name.upper()} Agent"]

    for col, df in enumerate([rb_df, rl_df]):
        hours = df["hour"].values

        # ---- Row 0: Power flows ----
        ax = fig.add_subplot(gs[0, col])
        ax.fill_between(hours, df["pv_w"], alpha=0.4,  color="#f39c12", label="PV Gen")
        ax.fill_between(hours, df["load_w"], alpha=0.4, color="#8e44ad", label="Load")
        ax.plot(hours, df["grid_w"], color="#c0392b", linewidth=1.5,
                linestyle="--", label="Grid Import")
        ax.set_title(f"{titles_row[col]} — Power Flows")
        ax.set_ylabel("Power (W)")
        ax.set_xlabel("Hour of Day")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)

        # ---- Row 1: Battery SOC ----
        ax = fig.add_subplot(gs[1, col])
        ax.plot(hours, df["soc"] * 100, color="#2980b9", linewidth=2)
        ax.axhline(20, color="red",    linestyle=":",  linewidth=1, label="Min SOC 20%")
        ax.axhline(95, color="orange", linestyle=":",  linewidth=1, label="Max SOC 95%")
        ax.fill_between(hours, df["soc"]*100, alpha=0.2, color="#2980b9")
        ax.set_title(f"{titles_row[col]} — Battery SOC")
        ax.set_ylabel("SOC (%)")
        ax.set_xlabel("Hour of Day")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # ---- Row 2: Actions ----
        ax = fig.add_subplot(gs[2, col])
        for i, (label, color) in enumerate(zip(ACTION_LABELS, ACTION_COLORS)):
            mask = df["action"] == i
            ax.scatter(df.loc[mask, "hour"], [i] * mask.sum(),
                       color=color, s=30, label=label, alpha=0.7)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(ACTION_LABELS)
        ax.set_title(f"{titles_row[col]} — Actions")
        ax.set_xlabel("Hour of Day")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Episode Energy Flow Comparison (One Week)", fontsize=13,
                 fontweight="bold", y=1.01)
    path = os.path.join(RESULTS_DIR, "episode_timeline.png")
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.close(fig)


# ------------------------------------------------------------------ #
#  Plot 3 – Action distribution                                       #
# ------------------------------------------------------------------ #
def plot_action_distribution(rb_df, rl_df, algo_name, save=True):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Action Distribution Comparison", fontsize=12, fontweight="bold")

    for ax, df, title in zip(axes, [rb_df, rl_df],
                              ["Rule-Based EMS", f"{algo_name.upper()} Agent"]):
        counts = [int((df["action"] == i).sum()) for i in range(4)]
        bars   = ax.bar(ACTION_LABELS, counts, color=ACTION_COLORS, width=0.5)
        ax.set_title(title)
        ax.set_ylabel("# Steps")
        for b, c in zip(bars, counts):
            ax.text(b.get_x() + b.get_width()/2, c + 0.5,
                    str(c), ha="center", va="bottom", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "action_distribution.png")
    if save:
        fig.savefig(path, dpi=150)
        print(f"Saved → {path}")
    plt.close(fig)


# ------------------------------------------------------------------ #
#  Print summary table                                                #
# ------------------------------------------------------------------ #
def print_summary_table(rb_metrics, rl_metrics, algo_name):
    improvement_cost = (
        (rb_metrics["mean_grid_cost"] - rl_metrics["mean_grid_cost"])
        / rb_metrics["mean_grid_cost"] * 100
    )
    improvement_ssc  = rl_metrics["mean_solar_ssc"] - rb_metrics["mean_solar_ssc"]

    print("\n" + "="*60)
    print(f"  COMPARISON RESULTS  (Rule-Based  vs  {algo_name.upper()})")
    print("="*60)
    print(f"{'Metric':<30} {'Rule-Based':>12} {algo_name.upper():>12}  {'Δ':>8}")
    print("-"*60)
    print(f"{'Mean Grid Cost ($)':<30} "
          f"{'${:.4f}'.format(rb_metrics['mean_grid_cost']):>12} "
          f"{'${:.4f}'.format(rl_metrics['mean_grid_cost']):>12}  "
          f"{improvement_cost:>+7.1f}%")
    print(f"{'Solar Self-Consumption (%)':<30} "
          f"{rb_metrics['mean_solar_ssc']:>11.1f}% "
          f"{rl_metrics['mean_solar_ssc']:>11.1f}%  "
          f"{improvement_ssc:>+7.1f}pp")
    print(f"{'Mean Episode Reward':<30} "
          f"{rb_metrics['mean_reward']:>12.3f} "
          f"{rl_metrics['mean_reward']:>12.3f}")
    print("="*60)
    if improvement_cost > 0:
        print(f"✓ RL reduces grid cost by {improvement_cost:.1f}%")
    else:
        print(f"✗ RL increases grid cost by {abs(improvement_cost):.1f}% "
              f"(may need more training)")
    if improvement_ssc > 0:
        print(f"✓ RL improves solar self-consumption by {improvement_ssc:.1f} pp")


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo",     default="dqn", choices=["dqn", "ppo"])
    parser.add_argument("--episodes", default=10,    type=int)
    parser.add_argument("--model_dir",default="models")
    args = parser.parse_args()

    algo = args.algo.lower()
    print(f"\nLoading {algo.upper()} model from '{args.model_dir}/'...")

    # Rule-based controller
    rb_ctrl = RuleBasedEMS()

    # Load RL model
    if SB3_AVAILABLE:
        try:
            ModelClass = DQN if algo == "dqn" else PPO
            rl_model   = ModelClass.load(
                os.path.join(args.model_dir, f"{algo}_microgrid")
            )
            print(f"Model loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Could not load model: {e}")
            print("  Run  python train_agent.py  first.")
            return
    else:
        print("stable_baselines3 not available — exiting.")
        return

    # Collect multi-episode metrics
    print(f"\nEvaluating over {args.episodes} episodes …")
    rb_metrics = collect_metrics(
        rb_ctrl, n_episodes=args.episodes, seed_start=200, is_rl=False
    )
    rl_metrics = collect_metrics(
        rl_model, n_episodes=args.episodes, seed_start=200, is_rl=True
    )

    # Single episode for timeline plots
    rb_df, _ = run_episode(rb_ctrl,   seed=42, is_rl=False)
    rl_df, _ = run_episode(rl_model,  seed=42, is_rl=True)

    # Generate all plots
    print("\nGenerating plots …")
    plot_comparison(rb_metrics, rl_metrics, algo)
    plot_episode_timeline(rb_df, rl_df, algo)
    plot_action_distribution(rb_df, rl_df, algo)

    # Print summary
    print_summary_table(rb_metrics, rl_metrics, algo)

    print(f"\nAll outputs saved to '{RESULTS_DIR}/' directory.")


if __name__ == "__main__":
    main()

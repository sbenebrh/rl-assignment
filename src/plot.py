"""
Generate reward-vs-steps learning curves from eval CSV logs.
Usage:
    python -m src.plot                  # plots all runs found in logs/
    python -m src.plot --runs 1 2 3     # explicit run IDs
    python -m src.plot --out plots/learning_curves.png
"""
import os
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
import csv


def load_csv(path):
    steps, rewards = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            rewards.append(float(row["mean_reward"]))
    return np.array(steps), np.array(rewards)


def plot_runs(run_ids, out_path):
    os.makedirs("plots", exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    final_rewards = []
    colors = ["tab:blue", "tab:orange", "tab:green"]

    for i, run_id in enumerate(run_ids):
        csv_path = f"logs/run{run_id}_eval.csv"
        if not os.path.exists(csv_path):
            print(f"  [warn] {csv_path} not found, skipping")
            continue

        steps, rewards = load_csv(csv_path)
        color = colors[i % len(colors)]

        ax.plot(steps, rewards, color=color, linewidth=1.5, label=f"Run {run_id}")
        ax.scatter(steps, rewards, color=color, s=15, zorder=5)

        final_rewards.append(rewards[-1])
        print(f"  Run {run_id}: final eval reward = {rewards[-1]:.1f} "
              f"| best = {rewards.max():.1f} @ step {steps[rewards.argmax()]:,}")

    if final_rewards:
        mean = np.mean(final_rewards)
        ax.axhline(mean, color="red", linestyle="--", linewidth=1.2,
                   label=f"Mean final reward: {mean:.1f}")
        print(f"\n  Average final reward across {len(final_rewards)} run(s): {mean:.1f}")

    ax.set_xlabel("Training steps", fontsize=13)
    ax.set_ylabel("Mean eval reward (unclipped)", fontsize=13)
    ax.set_title("DQN on BreakoutNoFrameskip-v4 — Learning Curves", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\n  Saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, nargs="+", default=None,
                        help="Run IDs to plot (default: auto-detect from logs/)")
    parser.add_argument("--out", type=str, default="plots/learning_curves.png")
    args = parser.parse_args()

    if args.runs:
        run_ids = args.runs
    else:
        # Auto-detect from logs/
        found = sorted(glob.glob("logs/run*_eval.csv"))
        run_ids = [int(p.split("run")[1].split("_")[0]) for p in found]
        if not run_ids:
            print("No eval CSV files found in logs/. Run training first.")
            exit(1)
        print(f"Auto-detected runs: {run_ids}")

    plot_runs(run_ids, args.out)

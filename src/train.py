import os
import csv
import argparse
import numpy as np
import torch
from src.preprocessing import make_env
from src.agent import DQNAgent

# ── Hyperparameters ──────────────────────────────────────────────────────────
# NOTE: these intentionally differ from the original paper (requirement #7).
# Key differences: Adam instead of RMSProp, Huber loss, shorter eps decay,
# smaller buffer (500k vs 1M), gradient clipping.

CONFIG = {
    "game":              "SeaquestNoFrameskip-v4",
    "buffer_size":       500_000,   # paper uses 1M, 250k is a memory-conscious compromise
    "batch_size":        256,       # larger batch to capture more dones/rewards (sparse signal)
    "gamma":             0.99,
    "lr":                0.0001,    # reduced to prevent Q-value explosion
    "eps_start":         1.0,
    "eps_end":           0.1,
    "eps_decay_steps":   250_000, # paper-scale decay — agent explores long enough to learn
    "eval_eps":          0.05,
    "learning_starts":   50_000,   # increased from 50k — more warmup for complex game
    "train_freq":        1,         # paper: update every action (not every 4)
    "eval_freq":         10_000,
    "eval_episodes":     10,         # fewer eval episodes for speed (was 10)
    "max_steps":         3_000_000, # extended from 3M — Seaquest needs more training
}


# ── Evaluation ───────────────────────────────────────────────────────────────

MAX_EVAL_STEPS = 5_000   # cap per episode — prevents eval hanging on long rallies

def evaluate(agent, n_episodes):
    """Run n_episodes with eval policy, return mean unclipped reward."""
    env = make_env(CONFIG["game"], episodic_life=False)
    rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False
        steps = 0
        while not done and steps < MAX_EVAL_STEPS:
            action = agent.select_action(state, eval_mode=True)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward          # unclipped — this is the real score
            done = terminated or truncated
            steps += 1
        rewards.append(episode_reward)
    env.close()
    return float(np.mean(rewards))


# ── Training loop ────────────────────────────────────────────────────────────

def train(run_id, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = make_env(CONFIG["game"])
    n_actions = env.action_space.n
    agent = DQNAgent(n_actions, CONFIG)

    print(f"\n{'='*55}")
    print(f"  Run {run_id} | seed={seed} | device={agent.device}")
    print(f"  Game: {CONFIG['game']} | Actions: {n_actions}")
    print(f"{'='*55}\n")

    # CSV log: append if resuming, else write fresh header
    log_path = f"logs/run{run_id}_eval.csv"
    resume_checkpoint = f"checkpoints/run{run_id}_best.pt"
    if os.path.exists(resume_checkpoint) and os.path.exists(log_path):
        agent.load(resume_checkpoint)
        total_steps = agent.total_steps
        best_reward = max(float(r) for _, r in (
            row.split(",") for row in open(log_path).read().strip().split("\n")[1:]
        ) if r)
        print(f"  Resuming from step {total_steps:,} | best so far: {best_reward:.1f}")
    else:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "mean_reward"])
        total_steps = 0
        best_reward = -float("inf")

    state, _ = env.reset()
    recent_losses = []  # Track losses for averaging

    while total_steps < CONFIG["max_steps"]:

        # 1. Select action (random during warm-up, ε-greedy after)
        action = agent.select_action(state)

        # 2. Step the environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_steps += 1

        # 3. Store transition with clipped reward (training only)
        clipped_reward = float(np.clip(reward, -1.0, 1.0))
        agent.store(state, action, clipped_reward, next_state, done)

        state = next_state if not done else env.reset()[0]

        # 4. Learn every train_freq steps (after warm-up)
        if total_steps % CONFIG["train_freq"] == 0:
            loss = agent.learn()
            if loss is not None:
                recent_losses.append(loss)

        # 5. Evaluate every eval_freq steps (assignment requirement)
        if total_steps % CONFIG["eval_freq"] == 0:
            mean_reward = evaluate(agent, CONFIG["eval_episodes"])
            eps = agent._current_eps()
            avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0.0
            print(
                f"  step={total_steps:>8,} | "
                f"eps={eps:.3f} | "
                f"loss={avg_loss:.4f} | "
                f"eval_reward={mean_reward:.1f}"
            )
            recent_losses = []  # Reset for next interval

            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([total_steps, mean_reward])

            if mean_reward > best_reward:
                best_reward = mean_reward
                agent.save(f"checkpoints/run{run_id}_best.pt")

    agent.save(f"checkpoints/run{run_id}_final.pt")
    env.close()
    print(f"\nRun {run_id} complete. Best eval reward: {best_reward:.1f}")
    return best_reward


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",  type=int, default=1,    help="Run ID: 1, 2, or 3")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: run_id * 42)")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else args.run * 42
    train(run_id=args.run, seed=seed)

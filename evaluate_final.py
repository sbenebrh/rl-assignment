#!/usr/bin/env python3
"""Evaluate a trained DQN agent on N episodes."""
import argparse
import numpy as np
from src.agent import DQNAgent
from src.preprocessing import make_env

# Same config as training
CONFIG = {
    "buffer_size": 500_000,
    "batch_size": 64,
    "gamma": 0.99,
    "lr": 0.0001,
    "eval_eps": 0.05,
}

MAX_EVAL_STEPS = 5_000


def evaluate(agent, game, n_episodes):
    """Run n_episodes with eval policy, return mean reward and all episode rewards."""
    env = make_env(game, episodic_life=False)
    rewards = []
    for i in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False
        steps = 0
        while not done and steps < MAX_EVAL_STEPS:
            action = agent.select_action(state, eval_mode=True)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            steps += 1
        rewards.append(episode_reward)
        print(f"  Episode {i+1}/{n_episodes}: {episode_reward:.0f}")
    env.close()
    return rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (e.g., checkpoints/run2_best.pt)")
    parser.add_argument("--episodes", type=int, default=30, help="Number of eval episodes")
    parser.add_argument("--game", type=str, default="SeaquestNoFrameskip-v4", help="Game name")
    args = parser.parse_args()

    print(f"\nEvaluating {args.checkpoint} on {args.episodes} episodes...")
    print("="*60)

    # Load agent
    env = make_env(args.game, episodic_life=False)
    n_actions = env.action_space.n
    env.close()

    agent = DQNAgent(n_actions, CONFIG)
    agent.load(args.checkpoint)

    # Evaluate
    rewards = evaluate(agent, args.game, args.episodes)

    # Stats
    print("="*60)
    print(f"\nResults ({args.episodes} episodes):")
    print(f"  Mean:   {np.mean(rewards):.1f}")
    print(f"  Median: {np.median(rewards):.1f}")
    print(f"  Std:    {np.std(rewards):.1f}")
    print(f"  Min:    {np.min(rewards):.1f}")
    print(f"  Max:    {np.max(rewards):.1f}")

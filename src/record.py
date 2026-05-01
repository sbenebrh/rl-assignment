"""
Record a video of a trained DQN agent playing Breakout.
Usage:
    python -m src.record --checkpoint checkpoints/run1_best.pt
    python -m src.record --checkpoint checkpoints/run1_best.pt --episodes 3 --out plots/
"""
import os
import argparse
import numpy as np
import torch
import cv2
from src.preprocessing import make_env
from src.agent import DQNAgent
from src.train import CONFIG


def record(checkpoint_path, n_episodes, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Use rgb_array render mode to capture frames
    env = make_env(CONFIG["game"])
    raw_env = env  # keep reference to access render

    # Rebuild env with render mode for recording
    import gymnasium as gym
    import ale_py
    gym.register_envs(ale_py)

    from src.preprocessing import MaxAndSkipEnv, WarpFrame, FrameStack
    render_env = gym.make(CONFIG["game"], render_mode="rgb_array")
    render_env = MaxAndSkipEnv(render_env, skip=4)
    render_env = WarpFrame(render_env)
    render_env = FrameStack(render_env, k=4)

    agent = DQNAgent(render_env.action_space.n, CONFIG)
    agent.load(checkpoint_path)
    agent.online_net.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Device: {agent.device}")

    for ep in range(n_episodes):
        state, _ = render_env.reset()
        frames = []
        episode_reward = 0.0
        done = False

        while not done:
            # Capture raw RGB frame before preprocessing
            raw_frame = render_env.env.env.env.render()  # unwrap to raw env
            frames.append(raw_frame)

            action = agent.select_action(state, eval_mode=True)
            state, reward, terminated, truncated, _ = render_env.step(action)
            episode_reward += reward
            done = terminated or truncated

        print(f"  Episode {ep + 1}: reward = {episode_reward:.1f} | frames = {len(frames)}")

        # Write video with OpenCV
        out_path = os.path.join(out_dir, f"breakout_ep{ep + 1}.mp4")
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,  # fps
            (w, h),
        )
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"  Saved → {out_path}")

    render_env.close()
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--episodes",  type=int, default=3,
                        help="Number of episodes to record")
    parser.add_argument("--out",       type=str, default="plots/",
                        help="Output directory for video files")
    args = parser.parse_args()

    record(args.checkpoint, args.episodes, args.out)

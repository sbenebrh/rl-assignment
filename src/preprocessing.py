import numpy as np
import gymnasium as gym
import ale_py
from collections import deque
import cv2

gym.register_envs(ale_py)


class MaxAndSkipEnv(gym.Wrapper):
    """Repeat action k times. Observe max over last 2 raw frames."""

    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
        self._frame_buffer = np.zeros((2, *env.observation_space.shape), dtype=np.uint8)

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False

        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._frame_buffer[i % 2] = obs
            total_reward += reward
            if terminated or truncated:
                break

        max_frame = np.max(self._frame_buffer, axis=0)
        return max_frame, total_reward, terminated, truncated, info


class WarpFrame(gym.ObservationWrapper):
    """Grayscale + resize to 84x84."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return np.expand_dims(resized, axis=2)


class FrameStack(gym.Wrapper):
    """Stack k last frames into a single observation."""

    def __init__(self, env, k=4):
        super().__init__(env)
        self._k = k
        self._frames = deque(maxlen=k)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self._k, 84, 84), dtype=np.uint8)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        arr = np.array(list(self._frames))
        return np.squeeze(arr, axis=-1)


def make_env(game="BreakoutNoFrameskip-v4"):
    """Compose all wrappers in the correct order."""
    env = gym.make(game)
    env = MaxAndSkipEnv(env, skip=4)
    env = WarpFrame(env)
    env = FrameStack(env, k=4)
    return env

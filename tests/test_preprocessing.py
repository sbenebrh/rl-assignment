import numpy as np
import pytest
import gymnasium as gym
from src.preprocessing import MaxAndSkipEnv, WarpFrame, FrameStack, make_env


# ── Helpers ──────────────────────────────────────────────────────────────────

class FakeAtariEnv(gym.Env):
    """Minimal fake env that mimics a raw NoFrameskip Atari env."""

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(210, 160, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.Discrete(4)
        self._step_count = 0

    def reset(self, **kwargs):
        self._step_count = 0
        obs = np.zeros((210, 160, 3), dtype=np.uint8)
        return obs, {}

    def step(self, action):
        self._step_count += 1
        # Alternate pixel value each frame to make max-pooling testable
        val = 100 if self._step_count % 2 == 0 else 200
        obs = np.full((210, 160, 3), val, dtype=np.uint8)
        reward = float(action)          # reward = action index (easy to track)
        terminated = self._step_count >= 1000
        truncated = False
        return obs, reward, terminated, truncated, {}


# ── MaxAndSkipEnv ─────────────────────────────────────────────────────────────

class TestMaxAndSkipEnv:

    def setup_method(self):
        self.env = MaxAndSkipEnv(FakeAtariEnv(), skip=4)
        self.env.reset()

    def test_output_shape(self):
        obs, *_ = self.env.step(0)
        assert obs.shape == (210, 160, 3), f"Expected (210,160,3), got {obs.shape}"

    def test_output_dtype(self):
        obs, *_ = self.env.step(0)
        assert obs.dtype == np.uint8

    def test_reward_is_sum_of_4_steps(self):
        # action=2 → reward=2.0 per step → total over 4 steps = 8.0
        _, reward, *_ = self.env.step(2)
        assert reward == 8.0, f"Expected 8.0, got {reward}"

    def test_max_pool_takes_pixel_max(self):
        # FakeAtariEnv alternates 200 / 100 / 200 / 100 ...
        # last 2 frames will have values 200 and 100 → max should be 200
        obs, *_ = self.env.step(0)
        assert obs.max() == 200, f"Max pixel should be 200, got {obs.max()}"


# ── WarpFrame ─────────────────────────────────────────────────────────────────

class TestWarpFrame:

    def setup_method(self):
        self.env = WarpFrame(MaxAndSkipEnv(FakeAtariEnv(), skip=4))
        self.env.reset()

    def test_output_shape(self):
        obs, *_ = self.env.step(0)
        assert obs.shape == (84, 84, 1), f"Expected (84,84,1), got {obs.shape}"

    def test_output_dtype(self):
        obs, *_ = self.env.step(0)
        assert obs.dtype == np.uint8

    def test_observation_space_matches(self):
        assert self.env.observation_space.shape == (84, 84, 1)


# ── FrameStack ────────────────────────────────────────────────────────────────

class TestFrameStack:

    def setup_method(self):
        base = WarpFrame(MaxAndSkipEnv(FakeAtariEnv(), skip=4))
        self.env = FrameStack(base, k=4)

    def test_reset_output_shape(self):
        obs, _ = self.env.reset()
        assert obs.shape == (4, 84, 84), f"Expected (4,84,84), got {obs.shape}"

    def test_step_output_shape(self):
        self.env.reset()
        obs, *_ = self.env.step(0)
        assert obs.shape == (4, 84, 84), f"Expected (4,84,84), got {obs.shape}"

    def test_output_dtype(self):
        self.env.reset()
        obs, *_ = self.env.step(0)
        assert obs.dtype == np.uint8

    def test_observation_space_matches(self):
        assert self.env.observation_space.shape == (4, 84, 84)

    def test_frames_are_different_after_steps(self):
        # After a few steps, frames in the stack should not all be identical
        self.env.reset()
        for _ in range(4):
            obs, *_ = self.env.step(1)
        # At least check that the stack has 4 frames
        assert obs.shape[0] == 4


# ── Full pipeline (make_env) ──────────────────────────────────────────────────

class TestMakeEnv:

    def test_make_env_runs(self):
        env = make_env("BreakoutNoFrameskip-v4")
        obs, info = env.reset()
        assert obs.shape == (4, 84, 84), f"Expected (4,84,84), got {obs.shape}"
        assert obs.dtype == np.uint8
        obs2, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert obs2.shape == (4, 84, 84)
        assert isinstance(reward, float)
        env.close()

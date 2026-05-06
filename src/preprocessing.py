import numpy as np
import gymnasium as gym
import ale_py
from collections import deque
import cv2

gym.register_envs(ale_py)


class NoopResetEnv(gym.Wrapper):
    """Take a random number (0..noop_max) of NOOPs on reset for stochasticity."""

    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self._noop_max = noop_max
        self._noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        n_noops = np.random.randint(1, self._noop_max + 1)
        for _ in range(n_noops):
            obs, _, terminated, truncated, info = self.env.step(self._noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class FireResetEnv(gym.Wrapper):
    """Press FIRE on reset — required by games like Breakout that need a
    button press to launch the ball. Seaquest does NOT need this (the sub
    appears immediately), so this wrapper is opt-in via make_env."""

    def __init__(self, env):
        super().__init__(env)
        meanings = env.unwrapped.get_action_meanings()
        assert meanings[1] == "FIRE", f"FIRE action expected at index 1, got {meanings}"
        assert len(meanings) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)  # FIRE
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)  # safety step
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class EpisodicLifeEnv(gym.Wrapper):
    """Treat life loss as terminal during training — improves credit assignment.
    The real game-over is still respected for episode-end logic."""

    def __init__(self, env):
        super().__init__(env)
        self._lives = 0
        self._real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._real_done = terminated or truncated
        # Check life loss in the underlying ALE
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self._lives:
            terminated = True   # signal life-loss as terminal to the learner
        self._lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        # Only do a real reset if the previous episode was a real game-over.
        # Otherwise step through the next life without resetting state.
        if self._real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, terminated, truncated, info = self.env.step(0)  # NOOP
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        self._lives = self.env.unwrapped.ale.lives()
        return obs, info


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


FIRE_ON_RESET_GAMES = ("Breakout", "SpaceInvaders", "Pong")


def _needs_fire_on_reset(game: str) -> bool:
    """Games that require pressing FIRE to start play.
    Seaquest, MsPacman, Asterix, etc. start automatically and should not get this wrapper."""
    return any(game.startswith(g) for g in FIRE_ON_RESET_GAMES)


def make_env(game="SeaquestNoFrameskip-v4", episodic_life=True, fire_on_reset=None):
    """Compose all wrappers in the correct order.

    Wrapper order matters:
      raw env → NoopReset → MaxAndSkip → EpisodicLife → [FireReset] → Warp → Stack

    fire_on_reset: if None, auto-detect from game name. Set explicitly to override.
    """
    env = gym.make(game)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if episodic_life:
        env = EpisodicLifeEnv(env)
    if fire_on_reset is None:
        fire_on_reset = _needs_fire_on_reset(game)
    if fire_on_reset:
        env = FireResetEnv(env)
    env = WarpFrame(env)
    env = FrameStack(env, k=4)
    return env

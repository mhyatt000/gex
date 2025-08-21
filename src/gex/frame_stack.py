import numpy as np
import gymnasium as gym
from collections import deque


class FrameStack(gym.Wrapper):
    """Environment wrapper that stacks the last ``num_stack`` observations."""

    def __init__(self, env: gym.Env, num_stack: int) -> None:
        super().__init__(env)
        if num_stack <= 0:
            raise ValueError("num_stack must be greater than 0")
        self.num_stack = num_stack
        self.frames: deque[np.ndarray] = deque(maxlen=num_stack)

        assert isinstance(env.observation_space, gym.spaces.Box)
        low = np.repeat(env.observation_space.low[None, ...], num_stack, axis=0)
        high = np.repeat(env.observation_space.high[None, ...], num_stack, axis=0)
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=env.observation_space.dtype
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames.clear()
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_observation(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self):
        assert len(self.frames) == self.num_stack
        return np.stack(self.frames, axis=0)

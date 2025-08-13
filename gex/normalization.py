import numpy as np
import gymnasium as gym


class NormalizeEnv(gym.Wrapper):
    """Environment wrapper that normalizes observations and actions.

    Actions passed to this wrapper are assumed to be normalized. They are
    unnormalized using the provided action statistics before being passed to the
    underlying environment. Observations returned from the environment are
    normalized before being returned to the caller.

    The wrapper expects optional mean and standard deviation vectors for both
    observations and actions. If provided, their shapes must match the
    corresponding space shapes of the environment.
    """

    def __init__(
        self,
        env: gym.Env,
        obs_mean: np.ndarray | None = None,
        obs_std: np.ndarray | None = None,
        action_mean: np.ndarray | None = None,
        action_std: np.ndarray | None = None,
    ) -> None:
        super().__init__(env)

        if (obs_mean is None) != (obs_std is None):
            raise ValueError("obs_mean and obs_std must both be provided or both be None")
        if (action_mean is None) != (action_std is None):
            raise ValueError(
                "action_mean and action_std must both be provided or both be None"
            )

        if obs_mean is not None:
            obs_mean = np.asarray(obs_mean, dtype=np.float32)
            obs_std = np.asarray(obs_std, dtype=np.float32)
            assert obs_mean.shape == env.observation_space.shape
            assert obs_std.shape == env.observation_space.shape

        if action_mean is not None:
            action_mean = np.asarray(action_mean, dtype=np.float32)
            action_std = np.asarray(action_std, dtype=np.float32)
            assert action_mean.shape == env.action_space.shape
            assert action_std.shape == env.action_space.shape

        self._obs_mean = obs_mean
        self._obs_std = obs_std
        self._action_mean = action_mean
        self._action_std = action_std

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self._obs_mean is not None:
            obs = self._normalize(obs, self._obs_mean, self._obs_std)
        return obs, info

    def step(self, action):
        if self._action_mean is not None:
            action = self._unnormalize(action, self._action_mean, self._action_std)
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self._obs_mean is not None:
            obs = self._normalize(obs, self._obs_mean, self._obs_std)
        return obs, reward, terminated, truncated, info

    def _unnormalize(self, x, mean, std):
        return x * std + mean

    def _normalize(self, x, mean, std):
        return (x - mean) / std

import numpy as np
import gymnasium as gym


class FlattenObservationWrapper(gym.ObservationWrapper):
    """Environment wrapper that flattens observations to 1D vectors.

    Observations returned by the wrapped environment are flattened using
    ``numpy.ravel`` before being passed to the caller. The observation space is
    adjusted accordingly.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError(
                "FlattenObservationWrapper only supports Box observation spaces"
            )
        low = np.asarray(env.observation_space.low).ravel()
        high = np.asarray(env.observation_space.high).ravel()
        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=env.observation_space.dtype,
        )

    def observation(self, observation):
        return np.asarray(observation).ravel()

import numpy as np
import gymnasium as gym


class ActionSequenceWrapper(gym.Wrapper):
    """Execute a flattened sequence of actions in the wrapped environment.

    The wrapped environment receives multiple actions at once. The action
    provided to :meth:`step` is expected to be a one-dimensional array that
    concatenates several actions for the underlying environment. The vector is
    reshaped according to ``env.action_space.shape`` and each action is executed
    sequentially. The observation from the last step is returned together with
    the cumulative reward and termination flags.

    Parameters
    ----------
    env:
        The environment to wrap.
    sequence_length:
        Number of actions to execute sequentially.
    """

    def __init__(self, env: gym.Env, sequence_length: int) -> None:
        super().__init__(env)
        self._sequence_length = sequence_length

        base_space = env.action_space
        base_shape = base_space.shape
        base_low = np.asarray(base_space.low).reshape(-1)
        base_high = np.asarray(base_space.high).reshape(-1)

        low = np.tile(base_low, sequence_length)
        high = np.tile(base_high, sequence_length)
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=base_space.dtype)

    def step(self, action):
        action = np.asarray(action).reshape(-1)
        base_shape = self.env.action_space.shape
        base_size = int(np.prod(base_shape))
        expected_size = self._sequence_length * base_size
        if action.size != expected_size:
            raise ValueError(
                f"Expected action of size {expected_size}, got {action.size}"
            )

        actions = action.reshape(self._sequence_length, *base_shape)

        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        obs = None
        for sub_action in actions:
            obs, reward, term, trunc, info = self.env.step(sub_action)
            total_reward += reward
            terminated = terminated or term
            truncated = truncated or trunc
            if term or trunc:
                break
        assert obs is not None
        return obs, total_reward, terminated, truncated, info

import gymnasium as gym


class SuccessInfoWrapper(gym.Wrapper):
    """Environment wrapper that renames ``info['is_success']`` to ``info['success']``.

    Some environments return an ``is_success`` flag in the ``info`` dictionary. This
    wrapper makes such environments compatible with interfaces expecting the key
    ``success`` instead by copying the value and removing the original key.
    """

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if "is_success" in info:
            info = dict(info)
            info["success"] = info.pop("is_success")
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if "is_success" in info:
            info = dict(info)
            info["success"] = info.pop("is_success")
        return obs, reward, terminated, truncated, info

import gymnasium as gym


class ContinuousTaskWrapper(gym.Wrapper):
    def __init__(self, env, max_episode_steps: int) -> None:
        super().__init__(env)
        self._elapsed_steps = 0
        self._max_episode_steps = max_episode_steps

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return super().reset(**kwargs)

    def step(self, action):
        ob, rew, terminated, truncated, info = super().step(action)
        self._elapsed_steps += 1
        truncated = self._elapsed_steps >= self._max_episode_steps
        info["TimeLimit.truncated"] = truncated
        return ob, rew, terminated, truncated, info


# A simple wrapper that adds a is_success key which SB3 tracks
class SuccessInfoWrapper(gym.Wrapper):
    def step(self, action):
        ob, rew, terminated, truncated, info = super().step(action)
        info["is_success"] = info["success"]
        return ob, rew, terminated, truncated, info

"""
Reward-component logging callback for SB3.

Reads per-step reward breakdown from the info dict (populated by
`compute_dense_reward()`'s `_cache_info`) and logs rolling averages
to the SB3 logger (TensorBoard + CSV).

Usage in sac_train.py:
    from reward_component_logger import RewardComponentCallback
    ...
    reward_cb = RewardComponentCallback(verbose=1)
    model.learn(..., callback=[checkpoint_callback, reward_cb])

This is additive-only: it does not modify any reward values or env behavior.
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class RewardComponentCallback(BaseCallback):
    """
    Logs per-step reward components as rolling averages to the SB3 logger.

    The environment's `compute_dense_reward()` stores individual components
    (info_exist_reward, approach_reward, grasp_reward, goal_reward,
    static_reward) in `self._cache_info`, which gets merged into the step
    `info` dict. VecMonitor passes these through in `self.locals["infos"]`.
    """

    COMPONENT_KEYS = [
        "info_exist_reward",
        "approach_reward",
        "grasp_reward",
        "goal_reward",
        "static_reward",
    ]

    def __init__(self, window_size: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.window_size = window_size
        self._buffers = {k: [] for k in self.COMPONENT_KEYS}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            for key in self.COMPONENT_KEYS:
                if key in info:
                    self._buffers[key].append(info[key])
                    # Keep only the last `window_size` values
                    if len(self._buffers[key]) > self.window_size:
                        self._buffers[key] = self._buffers[key][-self.window_size:]

        # Log rolling averages every time the model logs (roughly every
        # `train_freq` steps, controlled by SB3 internally)
        for key in self.COMPONENT_KEYS:
            if self._buffers[key]:
                mean_val = np.mean(self._buffers[key])
                self.logger.record(f"reward/{key}", mean_val)

        return True

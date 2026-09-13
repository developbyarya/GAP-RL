import os
import json
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class DeterministicEvalCallback(BaseCallback):
    """
    Evaluates the policy deterministically on a separate eval_env.
    Logs eval/success_rate and eval/success_rate_once.
    Saves the best model based on eval/success_rate.
    """
    def __init__(self, eval_env, eval_freq, n_eval_episodes=20, best_model_save_path=None, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_model_save_path = best_model_save_path
        self.best_success_rate = -1.0
        
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            successes = []
            successes_once = []
            
            # Since eval_env is a VecEnv (SubprocVecEnv), we reset it
            obs = self.eval_env.reset()
            episodes_completed = 0
            
            while episodes_completed < self.n_eval_episodes:
                # Deterministic=True disables gSDE noise (returns mean action)
                action, _ = self.model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = self.eval_env.step(action)
                
                for i, done in enumerate(dones):
                    if done:
                        episodes_completed += 1
                        info = infos[i]
                        successes.append(info.get("is_success", 0.0))
                        successes_once.append(info.get("is_success_once", 0.0))
                        
                        if episodes_completed >= self.n_eval_episodes:
                            break
            
            mean_success = np.mean(successes)
            mean_success_once = np.mean(successes_once)
            
            self.logger.record("eval/success_rate", mean_success)
            self.logger.record("eval/success_rate_once", mean_success_once)
            
            if self.verbose > 0:
                print(f"Eval at {self.num_timesteps} steps: success_rate={mean_success:.3f}, success_rate_once={mean_success_once:.3f}")
            
            # Save best model
            if mean_success > self.best_success_rate:
                self.best_success_rate = mean_success
                if self.best_model_save_path is not None:
                    os.makedirs(self.best_model_save_path, exist_ok=True)
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                    
                    record_file = os.path.join(self.best_model_save_path, "best_model_record.json")
                    record_data = {
                        "best_step": self.num_timesteps,
                        "best_success_rate": mean_success,
                        "best_success_rate_once": mean_success_once
                    }
                    with open(record_file, "w") as f:
                        json.dump(record_data, f, indent=4)
                        
                    if self.verbose > 0:
                        print(f"New best model saved! (success_rate: {mean_success:.3f})")
                        
        return True

import torch
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback

class PlasticityResetCallback(BaseCallback):
    """
    Implements 'Shrink and Perturb' (Ash & Adams) for plasticity injection.
    Instead of a hard reset to 0 (which can cause catastrophic forgetting if the buffer
    is overly reliant on the old Q-values), this safely shrinks the weights by 0.5 
    and adds a tiny bit of noise to re-awaken dead neurons while preserving knowledge.
    """
    def __init__(self, reset_freq: int, verbose: int = 1):
        super().__init__(verbose)
        self.reset_freq = reset_freq
        self.shrink_factor = 0.5
        self.noise_std = 0.01

    def _shrink_and_perturb(self, layer):
        if isinstance(layer, nn.Linear):
            layer.weight.data = layer.weight.data * self.shrink_factor + torch.randn_like(layer.weight.data) * self.noise_std
            if layer.bias is not None:
                layer.bias.data = layer.bias.data * self.shrink_factor + torch.randn_like(layer.bias.data) * self.noise_std

    def _on_step(self) -> bool:
        if self.num_timesteps > 0 and self.num_timesteps % self.reset_freq == 0:
            if self.verbose > 0:
                print(f"\n[Step {self.num_timesteps}] Triggering Shrink and Perturb Plasticity Injection...")
            
            model = self.model
            
            with torch.no_grad():
                # Shrink Critic
                for q_net in model.critic.q_networks:
                    self._shrink_and_perturb(q_net[-1])

                # Shrink Actor
                if hasattr(model.actor, 'mu'):
                    self._shrink_and_perturb(model.actor.mu)
                if hasattr(model.actor, 'log_std'):
                    self._shrink_and_perturb(model.actor.log_std)

                # Sync target critic
                for param, target_param in zip(model.critic.parameters(), model.critic_target.parameters()):
                    target_param.data.copy_(param.data)
                
            # Clear optimizer momentum
            if hasattr(model.actor.optimizer, 'state'):
                model.actor.optimizer.state.clear()
            if hasattr(model.critic.optimizer, 'state'):
                model.critic.optimizer.state.clear()
            
            if self.verbose > 0:
                print(f" -> Network safely shrunk by {self.shrink_factor}x and momentum cleared.")
            
        return True

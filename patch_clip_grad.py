import re

def patch():
    with open('gap_rl/algorithms/scripts/custom_sac.py', 'r') as f:
        content = f.read()

    # Critic update
    old_critic_opt = """            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()"""
    new_critic_opt = """            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            if getattr(self.critic, "use_attn_lstm", False):
                th.nn.utils.clip_grad_norm_(self.critic.lstm.parameters(), max_norm=1.0)
                th.nn.utils.clip_grad_norm_(self.critic.token_attn.parameters(), max_norm=1.0)
            self.critic.optimizer.step()"""
    content = content.replace(old_critic_opt, new_critic_opt)

    # Actor update
    old_actor_opt = """            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()"""
    new_actor_opt = """            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            if getattr(self.actor, "use_attn_lstm", False):
                th.nn.utils.clip_grad_norm_(self.actor.lstm.parameters(), max_norm=1.0)
                th.nn.utils.clip_grad_norm_(self.actor.token_attn.parameters(), max_norm=1.0)
            self.actor.optimizer.step()"""
    content = content.replace(old_actor_opt, new_actor_opt)

    with open('gap_rl/algorithms/scripts/custom_sac.py', 'w') as f:
        f.write(content)

patch()
print("Patched gradient clipping")

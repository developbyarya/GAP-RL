def patch():
    with open('gap_rl/algorithms/scripts/custom_sac.py', 'r') as f:
        content = f.read()
    
    old_critic = """            if getattr(self.critic, "use_attn_lstm", False):
                th.nn.utils.clip_grad_norm_(self.critic.lstm.parameters(), max_norm=5.0)
                th.nn.utils.clip_grad_norm_(self.critic.token_attn.parameters(), max_norm=5.0)"""
    new_critic = """            if getattr(self.critic, "use_attn_lstm", False):
                th.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)"""
    content = content.replace(old_critic, new_critic)
    
    old_actor = """            if getattr(self.actor, "use_attn_lstm", False):
                th.nn.utils.clip_grad_norm_(self.actor.lstm.parameters(), max_norm=5.0)
                th.nn.utils.clip_grad_norm_(self.actor.token_attn.parameters(), max_norm=5.0)"""
    new_actor = """            if getattr(self.actor, "use_attn_lstm", False):
                th.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)"""
    content = content.replace(old_actor, new_actor)

    with open('gap_rl/algorithms/scripts/custom_sac.py', 'w') as f:
        f.write(content)

patch()
print("Patched wide clipping")

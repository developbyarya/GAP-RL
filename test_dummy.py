class Actor:
    def __init__(self, use_sde=False, log_std_init=-3):
        self.log_std_init = log_std_init
        
class CustomActor(Actor):
    def __init__(self, use_sde=False, log_std_init=-3, extra=None):
        super().__init__(use_sde, log_std_init)
        
class SACPolicy:
    def __init__(self, use_sde=False, log_std_init=-3):
        self.actor_kwargs = {"use_sde": use_sde, "log_std_init": log_std_init}
        self.make_actor()
        
    def make_actor(self):
        self.actor = Actor(**self.actor_kwargs)

class CustomSACPolicy(SACPolicy):
    def __init__(self, orig=None, use_sde=False, log_std_init=-3):
        super().__init__(use_sde, log_std_init)
        
    def make_actor(self):
        kwargs = self.actor_kwargs.copy()
        kwargs["extra"] = 7
        self.actor = CustomActor(**kwargs)
        
class BaseAlgorithm:
    def __init__(self, policy_class, policy_kwargs=None):
        self.policy_class = policy_class
        self.policy_kwargs = policy_kwargs or {}
        self.policy = self.policy_class(**self.policy_kwargs)
        
algo = BaseAlgorithm(CustomSACPolicy, policy_kwargs={"log_std_init": -1.5, "orig": "hi"})
print("Actor log_std_init:", algo.policy.actor.log_std_init)

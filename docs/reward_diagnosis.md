# Reward Ablation Diagnosis Results

We compared two training runs of ~300k timesteps to diagnose the `info_exist_reward` collapse hypothesis.

### Run: Control (info-exist-weight = 3.0)
- **train/std (first 50k steps)**: 0.0262
- **train/std (overall)**: 0.0290
- **rollout/success_rate (max)**: 0.00%
- **rollout/success_rate (final)**: 0.00%
- **reward/info_exist_reward**: 2.523 (40.0%)
- **reward/approach_reward**: 3.782 (60.0%)
- **reward/grasp_reward**: 0.001 (0.0%)
- **train/ent_coef (overall)**: 0.02427
### Run: Test (info-exist-weight = 0.3)
- **train/std (first 50k steps)**: 0.0262
- **train/std (overall)**: 0.0291
- **rollout/success_rate (max)**: 0.00%
- **rollout/success_rate (final)**: 0.00%
- **reward/info_exist_reward**: 0.229 (5.4%)
- **reward/approach_reward**: 4.002 (93.8%)
- **reward/grasp_reward**: 0.028 (0.6%)
- **train/ent_coef (overall)**: 0.02280

---

## Conclusion & Interpretation
1. **Hypothesis Refuted**: Reducing the `info_exist_reward` weight from 3.0 to 0.3 **did not** prevent the `train/std` collapse. The actor's standard deviation still pinned at ~0.029 in both runs, and the success rate remained 0.00%. This proves the cheap local optimum was NOT the primary cause of the premature exploration collapse.
2. **Entropy Coefficient Collapse**: The `train/ent_coef` dropped sharply from its initialization of 0.2 down to ~0.023 in both runs. The target entropy calculation (`target_entropy = -7.0`) or the tuning mechanism is failing to keep the entropy coefficient high enough to force exploration early on.
3. **gSDE Initialization**: Since we are using State-Dependent Exploration (`use_sde=True`), the `log_std_init=-3.67` forces the initial std to be extremely small (~0.025). The entropy tuning is unable to overcome this because the SDE noise matrix is not tied directly to per-step variance the way standard Gaussian noise is.
4. **Behavioral shift**: Even though it didn't solve the collapse, reducing the weight *did* shift the reward landscape. The `reward/approach_reward` became the dominant component (93.8% instead of 60.0%), and we saw a slight uptick in `reward/grasp_reward` (from 0.001 to 0.028), meaning the agent explored contact slightly more. 

### Recommended Next Steps
- **Do NOT** permanently merge the `info_exist_weight=0.3` change just yet, as it doesn't fix the core issue.
- **Investigate gSDE parameters**: We should test changing `log_std_init=-3.67` to a much larger value like `-1.0` or `0.0` to force larger initial exploration.
- **Investigate Aux Loss dominance**: The `aux_weight` and `target_weight` start at 100. It's highly likely that these auxiliary losses are dominating the critic/actor gradients, preventing the RL reward from guiding the policy early in training.

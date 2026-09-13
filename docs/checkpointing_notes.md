# Checkpointing and Evaluation Notes

## 1. Current Checkpoint Saving Behavior (Before Modification)
In the original `sac_train.py`, SB3's `CheckpointCallback` was used:
```python
checkpoint_callback = CheckpointCallback(
    save_freq=400000 // cfg.get("train_procs", 1), 
    save_path=log_dir
)
```
This saved checkpoints every 400,000 steps. There was **no** in-training evaluation callback running during the rollout loop. The script only ran a post-training evaluation loop over all saved checkpoints at the very end of `sac_train.py`.

## 2. Modifications Made
1. **Checkpoint Granularity:** The `save_freq` for `CheckpointCallback` has been modified to `200000 // train_procs` so a checkpoint is emitted every 200,000 steps, improving granularity without drowning the disk.
2. **In-Training Deterministic Evaluation:**
   - Injected a dedicated 1-process `SubprocVecEnv` in `sac_train.py` named `eval_vec_env` that perfectly mimics the training environment state (including `FrameStackWrapper`).
   - Created `custom_eval_callback.py` with `DeterministicEvalCallback`. 
   - Every 200,000 steps, it pauses training, calls `eval_env.reset()`, and runs exactly 20 episodes. 
3. **Deterministic gSDE Execution:**
   - In `CustomActor.forward`, the `deterministic` flag is correctly propagated to `self.action_dist.actions_from_params(...)`. 
   - SB3's `StateDependentNoiseDistribution` natively handles `deterministic=True` by yielding the un-noised mean action. Passing `deterministic=True` to `model.predict(...)` in the callback is fully sufficient.
4. **Best Checkpoint Tracking:**
   - The callback logs `eval/success_rate` and `eval/success_rate_once` directly to the active SB3 logger. These will appear in `progress.csv` seamlessly inline with the active step count.
   - The callback actively tracks the highest observed `eval/success_rate`. When beaten, the active model is exported to `best_checkpoint/best_model.zip` inside the run directory.
   - It also populates a `best_checkpoint/best_model_record.json` containing the step count and precise metrics so you don't have to parse TensorBoard yourself.

## 3. Exact Remote Command
Run this command from the `GAP-RL` directory on the remote server to launch the fixed, fully instrumented LSTM training run. It restores all the "best-known-good" parameters (info_exist=0.3, auto_0.2 ent_coef, UTD=1, n_stack=4).

```bash
# Launch background training with nohup or screen/tmux
python3 gap_rl/algorithms/scripts/sac_train.py \
  --cfg gap_rl/algorithms/config/egopoints_ur85_bezier2d_goalaux.yaml \
  --use-lstm
```
*(Note: `custom_sac.py` defaults to `use_lstm=False` if not specified, so ensure `--use-lstm` is explicitly passed).*

### What to check during the run:
- Look at `runs/<your_run>/progress.csv`. You should periodically see rows populating `eval/success_rate` and `eval/success_rate_once` every ~200k steps.
- Look at `runs/<your_run>/`. You should see `rl_model_200000_steps.zip`, `rl_model_400000_steps.zip`, etc., appearing periodically.
- Look inside `runs/<your_run>/best_checkpoint/`. You should see `best_model.zip` and a `best_model_record.json`. `cat` the JSON file to see what step the best model came from!

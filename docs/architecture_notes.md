# Architecture Notes

## Step 1: LSTM Hidden State Handling
The LSTM's hidden state is **not** persisted or stored alongside replay buffer transitions. It is always recomputed from scratch over a fixed `n_stack`-frame window. `FrameStackWrapper` stacks raw observations, and at sample time, `CustomActor._extract_windowed_features` runs the feature extractor on each frame in the window independently, then feeds the resulting sequence to the LSTM, taking only the final output step.

## Step 2: Episode-Start Data Quality
Because the LSTM hidden state is not persisted, we do not suffer from the "stale hidden state from an old policy" problem directly. However, we do have a related issue: `_init_history()` seeds a fresh episode's `n_stack`-frame window by repeating the first real observation `n_stack` times. For the first `n_stack - 1` steps of every episode, the LSTM's input window is degenerate — it contains identical repeated frames rather than genuine temporal history. Any replay-buffer transition sampled from near an episode boundary has this corrupted temporal context, which degrades the gradient signal at sequence boundaries.

## Step 5: Burn-in-style Loss Masking Scope
The mitigation implemented (`--mask-early-episode-loss`) is a scoped, tractable version of "burn-in" for this specific codebase. Instead of a full R2D2-style sequential replay buffer with persisted hidden states and a burn-in prefix, we simply exclude/down-weight actor/critic loss contributions from transitions sampled from the first `n_stack - 1` steps of an episode. Since this codebase doesn't store hidden states in the buffer at all, this scoping avoids a massive rearchitecture while directly addressing the degenerate-padding issue.

## Step 6: Vanilla GAP-RL Baseline
The original, unmodified GAP-RL baseline policy class (no LSTM, no frame stacking) is the standard Stable Baselines 3 `SAC` with a `MultiInputPolicy`. This config doesn't need to be reconstructed from history—we can simply instantiate `stable_baselines3.SAC` rather than `CustomSAC` for the vanilla baseline evaluation.

## Step 7: Execution Commands and Decision Criteria

**Exact Commands:**
1. **Sanity-check training run (~20k steps):**
   ```bash
   python -m gap_rl.algorithms.scripts.sac_train --use-grasp-tracking --total-timesteps 20000
   ```
2. **Full training run (matched budget):**
   ```bash
   python -m gap_rl.algorithms.scripts.sac_train --use-grasp-tracking --total-timesteps <BUDGET>
   ```
3. **Three-way eval comparison:**
   ```bash
   # (a) Current best T-GAP-RL checkpoint (no tracking)
   python -m gap_rl.algorithms.scripts.eval_tgaprl --checkpoint <PATH_A>
   
   # (b) New checkpoint with Grasp Tracking enabled
   python -m gap_rl.algorithms.scripts.eval_tgaprl --checkpoint <PATH_B> --use-grasp-tracking
   
   # (c) Plain vanilla GAP-RL baseline
   python -m gap_rl.algorithms.scripts.eval_tgaprl --checkpoint <PATH_C> --vanilla-baseline
   ```

**Decision Criteria:**
| Signal | Supports adopting Grasp Tracking |
| --- | --- |
| Success rate vs. current best (no tracking) checkpoint, same budget | Higher or equal |
| Vanilla GAP-RL baseline eval (Step 6c) vs. its own historical numbers | Unchanged (confirms nothing upstream broke) |
| Cold-start / track-loss frequency (log these explicitly) | Low, and not correlated with failed episodes |
| `--mask-early-episode-loss` on vs. off, same tracking config | Report both; adopt only if it clearly helps, since this is an untested mitigation, not a guaranteed one |

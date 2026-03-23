# RL Training Results — Pick-and-Place (PyBullet)
**Date:** 2026-03-23
**Algorithms:** SAC, TD3, DDPG via Stable-Baselines3

---

## What We Built

A full reinforcement learning pipeline for a robotic pick-and-place task:

1. **Environment** (`pick_place_env.py`) — A custom Gymnasium-compatible wrapper around PyBullet. A Kuka 7-DoF robot arm must learn to pick up a cube and place it in a tray.
2. **Training script** (`train.py`) — Plug-and-play support for SAC, TD3, and DDPG.

### MDP Definition

| Component | Description |
|---|---|
| **State** (23-dim) | 7 joint angles + 7 joint velocities + EE position (3) + cube position (3) + tray position (3) |
| **Action** (7-dim) | Joint position targets, clipped to [-1, 1] rad |
| **Reward** | `-dist(EE→cube) - dist(cube→tray)` + grasp bonus (+10) + success bonus (+50) |
| **Episode ends** | Successful placement OR 500 steps exceeded |

---

## Algorithm Comparison

All three algorithms trained for **100,000 timesteps** on the same environment and same GPU.

| Algorithm | Final Reward | Best Eval Reward | Runtime | Speed |
|---|---|---|---|---|
| **SAC** | -460 | **-416 ± 72** | 37 min 45 sec | 44 it/s |
| **TD3** | -466 | -482 ± 60 | 19 min 19 sec | 110 it/s |
| **DDPG** | -484 | -623 ± 178 | 20 min 43 sec | 100 it/s |

---

## Per-Algorithm Results

### SAC (Soft Actor-Critic)

| Checkpoint | Mean Reward |
|---|---|
| Step 0 (start) | -902 |
| Step 5,000 | -855 ± 113 |
| Step 50,000 | ~-500 |
| Step 95,000 | -433 ± 75 |
| Step 100,000 | **-416 ± 72** |

**Improvement: ~54%**

---

### TD3 (Twin Delayed DDPG)

| Checkpoint | Mean Reward |
|---|---|
| Step 0 (start) | ~-900 |
| Step 100,000 | **-482 ± 60** |

**Improvement: ~46%** — fastest training (110 it/s), most stable variance (±60)

---

### DDPG (Deep Deterministic Policy Gradient)

| Checkpoint | Mean Reward |
|---|---|
| Step 0 (start) | ~-900 |
| Step 100,000 | **-623 ± 178** |

**Improvement: ~31%** — weakest performance, highest variance (±178)

---

## What the Results Mean

- **Reward is negative** because it is purely distance-based — the agent is penalized for how far the end-effector is from the cube, and how far the cube is from the tray. A reward of 0 would mean perfect placement.
- **SAC performed best** — entropy regularization helps it explore more effectively in continuous action spaces.
- **TD3 was fastest and most stable** — its twin critic architecture reduces overestimation, making it reliable even with fewer steps.
- **DDPG struggled most** — it is the oldest and simplest of the three; without entropy regularization or twin critics, it tends to overfit early suboptimal behaviors.
- **No successful placements yet** across any algorithm — the +50 success bonus was never triggered. The agents are learning approach behavior. Actual grasping typically emerges at 300k–1M steps for a 7-DoF arm.
- **Episode length stayed at 500** throughout — no agent solved the task within the step limit.

---

## Saved Files

```
results/
├── models/
│   ├── sac_pick_place/final.zip            # SAC final model
│   ├── sac_pick_place_best/best_model.zip  # SAC best eval (-416)
│   ├── sac_pick_place_checkpoints/         # SAC checkpoints every 10k steps
│   ├── td3_pick_place/final.zip            # TD3 final model
│   ├── td3_pick_place_best/best_model.zip  # TD3 best eval (-482)
│   ├── ddpg_pick_place/final.zip           # DDPG final model
│   └── ddpg_pick_place_best/best_model.zip # DDPG best eval (-623)
├── logs/
│   ├── sac_run.log
│   ├── td3_run.log
│   └── ddpg_run.log
└── RESULTS.md                              # this file
```

---

## Next Steps

1. **Train longer** — Run 500k–1M steps to see actual grasping emerge
2. **SAC is the winner so far** — prioritise tuning SAC hyperparameters (learning rate, entropy coefficient)
3. **Improve reward shaping** — Add orientation reward, gripper contact signal
4. **Use a gripper** — Switch to a robot with an actual gripper (e.g. Franka Panda + PyBullet)

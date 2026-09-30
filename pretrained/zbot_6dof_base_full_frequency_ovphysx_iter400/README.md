# ZBot 6-DoF full-frequency Base checkpoint, iteration 400

Source run: `logs/rsl_rl/zbot_6dof_base/2026-09-24_21-38-58_full_frequency_from_300`

- Physics: OvPhysX
- Frequency range: 0.5–2.0 Hz, sampled uniformly over the full range
- Source checkpoint: `model_400.pt`
- Walking initialization: `walking_actor_prior.pt`
- Periodic walking continuation: `periodic_walking_actor_prior.pt`
- Transferred state: the first 39 semantically shared observation columns and all later actor MLP layers
- Reset state: walking-only input columns, critic, optimizer, iteration, and exploration standard deviation

Fixed-frequency evaluation of `model_400.pt`:

| Frequency | COM score | Force score | COM correlation | Force correlation |
| --- | ---: | ---: | ---: | ---: |
| 0.5 Hz | 0.5130 | 0.2054 | 0.9909 | 0.8828 |
| 1.0 Hz | 0.9225 | 0.2086 | 0.9928 | 0.8863 |
| 1.5 Hz | 0.7064 | 0.2186 | 0.9800 | 0.8697 |
| 2.0 Hz | 0.4387 | 0.2221 | 0.9526 | 0.8390 |

Use `walking_actor_prior.pt` with `scripts/train_compact.py --actor_prior`; do not load the Base checkpoint
directly into the walking task because the two 44-dimensional observations have different meanings in columns 39–43.

Use `periodic_walking_actor_prior.pt` with `ZbotRlIsaaclab-6DOF-Periodic-Walking`. That task preserves the
entire Base observation and frequency/phase contract, so all 44 actor input columns transfer unchanged while the
full walking reward set adds forward speed, step length, step symmetry, heading/yaw, contact, and joint regularization.

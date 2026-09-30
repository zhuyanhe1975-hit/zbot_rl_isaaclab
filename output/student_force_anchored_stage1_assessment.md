# Student force-pulse PPO, stage 1: 64-environment assessment

Run: `logs/rsl_rl/zbot_6dof_periodic_walking_student_robust/2026-09-27_16-22-11_student_force_anchored_stage1`.
Baseline: `bests/zbot_6dof_periodic_walking_student_quality_20260927_model374/model_374.pt`.

Each evaluation used 64 ordered frequencies spanning 0.5–2.0 Hz, 1000 policy steps, and the same evaluation seed. Disturbed evaluation enabled startup friction/mass/actuator variation and full-strength 0.16 s physical base-force pulses. The baseline received 146 pulses across 64 environments; model 25 received 168.

| Checkpoint | Evaluation | Mean episode steps | Environments with a fall | Mean absolute frequency error |
| --- | --- | ---: | ---: | ---: |
| Archived student 374 | Nominal | 968.8 | 4/64 | 0.081 Hz |
| Archived student 374 | Disturbed | 792.7 | 23/64 | 0.120 Hz |
| Robust model 25 | Nominal | 919.3 | 9/64 | 0.093 Hz |
| Robust model 25 | Disturbed | 867.7 | 15/64 | 0.111 Hz |
| Robust model 50 | Disturbed | 843.5 | 17/64 | 0.118 Hz |
| Robust model 75 | Disturbed | 786.2 | 25/64 | 0.112 Hz |
| Robust model 99 | Nominal | 893.2 | 12/64 | 0.093 Hz |
| Robust model 99 | Disturbed | 769.7 | 25/64 | 0.125 Hz |

Model 25 shows a real tradeoff: better recovery under the tested force distribution, with worse nominal survival and cadence. In the 0.5–0.85 Hz group, its disturbed mean episode length rose from 600 to 736 steps, while its nominal frequency error rose from 0.112 to 0.143 Hz. Later checkpoints did not improve this balance. Keep the archived student as the nominal baseline; model 25 is an experimental disturbance-specialized candidate.

These are single-seed, 20-second evaluations. They do not establish performance across other seeds, longer runs, or stronger forces.

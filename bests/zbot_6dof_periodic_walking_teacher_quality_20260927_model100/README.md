# Quality teacher — model 100

Selected checkpoint from `logs/rsl_rl/zbot_6dof_periodic_walking_teacher_quality/2026-09-27_13-18-21_teacher_quality`.

- Task: `ZbotRlIsaaclab-6DOF-Periodic-Walking-Teacher-Quality`
- Physics: Isaac Sim PhysX
- Checkpoint: `model_100.pt` (zero-based iteration 100)
- `params/`: saved training environment and agent configuration
- `training.tfevents`: TensorBoard scalars from the source run
- `evaluation_64env.csv`: independent 1000-step playback at 64 ordered frequencies from 0.5 to 2.0 Hz

All 64 evaluation environments ran for 1000 steps. In the 0.5–0.85 Hz group, the mean absolute frequency error was 0.103 Hz; mean heading error was 0.366 rad. The average of each environment's P90 touchdown force was 266.4 N and P90 step length was 0.379 m. Model 299 reduced low-frequency force slightly but increased heading error, so model 100 is the balanced teacher selection.

The evaluation used the current touchdown detector, which filters contacts within 0.15 s of the previous accepted touchdown. The saved `params/env.yaml` records the original training configuration, before that detector correction.

From the project root:

```bash
./evaluate_student.sh bests/zbot_6dof_periodic_walking_teacher_quality_20260927_model100/model_100.pt quality_teacher 1000 64
```

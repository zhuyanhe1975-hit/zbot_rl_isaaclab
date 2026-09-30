# Quality student — model 374

Selected checkpoint from `logs/rsl_rl/zbot_6dof_periodic_walking_student_quality/2026-09-27_14-35-20_student_blended_pilot`.

- Task: `ZbotRlIsaaclab-6DOF-Periodic-Walking-Student-Quality`
- Physics: Isaac Sim PhysX
- Checkpoint: `model_374.pt` (zero-based iteration 374)
- Lineage: distilled from the selected quality teacher, then resumed from the original student's `model_275.pt` using blended teacher/student actions and a 1e-4 learning rate
- `params/`: saved training environment and agent configuration
- `training.tfevents`: TensorBoard scalars from the resumed source run
- `evaluation_64env.csv`: student-only 1000-step playback at 64 ordered frequencies from 0.5 to 2.0 Hz

In the 0.5–0.85 Hz group, the mean absolute frequency error was 0.112 Hz, down from 0.196 Hz for the pre-resume student model 275. Mean heading error was 0.363 rad; the averages of each environment's P90 swing height, touchdown force, and step length were 0.186 m, 258.8 N, and 0.377 m. Four of the 64 environments fell during playback. This is a distillation checkpoint for later nominal fine-tuning and disturbance training, not a final robust policy.

The resumed training run restarted the frequency curriculum. Independent evaluation still covered the full 0.5–2.0 Hz range. Subsequent `train_student.sh --resume` runs now disable that curriculum restart.

From the project root:

```bash
./evaluate_student.sh bests/zbot_6dof_periodic_walking_student_quality_20260927_model374/model_374.pt quality_student 1000 64
./run_student_quality.sh range bests/zbot_6dof_periodic_walking_student_quality_20260927_model374/model_374.pt
```

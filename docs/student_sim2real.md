# Periodic walking student: sim-to-real contract

The archived teacher at `bests/zbot_6dof_periodic_walking_isaacsim_physx_20260926_iter1500/model_1499.pt`
is the actor-only starting point for the new quality teacher. Select a fully trained quality-teacher
checkpoint before distillation. Its policy consumes 45 simulation observations. The student is a 38-input GRU policy and never receives
the simulated base linear velocity, foot contact forces, whole-body COM, or support-distance error.

## Student inputs at 50 Hz

| Columns | Signal | Real source | Units / processing |
| --- | --- | --- | --- |
| 0–2 | `base_ang_vel` | IMU gyro | rad/s in `base` axes |
| 3–5 | `projected_gravity` | IMU orientation | unit gravity vector in `base` axes |
| 6 | `heading_error` | IMU yaw estimate | wrapped startup yaw minus current yaw, rad |
| 7–9 | `base_lin_acc` | IMU accelerometer | m/s², including gravity, multiplied by 0.1 |
| 10–15 | `joint_pos` | six joint encoders | rad, relative to the saved default pose |
| 16–21 | `joint_vel` | six joint velocity estimates | rad/s |
| 22–27 | `joint_effort` | MIT-mode motor torque feedback | N·m multiplied by 0.01 |
| 28–33 | previous policy action | controller memory | raw network output, before `tanh` |
| 34–37 | frequency/support/phase command | controller clock | Hz, ±1, sine, cosine |

The observation order is fixed by `StudentPolicyCfg`. The teacher's separate `teacher` group retains the
original 45-input order only during distillation. Student sensor noise is enabled during training; teacher
label observations are clean. The student uses a GRU so recent encoder, IMU, and torque history can help
infer the hidden motion and contact state. Isaac Lab's PhysX `Imu` model biases its accelerometer with
gravity, matching the real sensor convention. Simulated yaw comes from the base orientation as a surrogate
for a real IMU orientation estimate, with the same heading sign as the teacher. The deployment controller
zeros yaw at trial start. `zbot_rl_isaaclab.deployment.StudentControllerState` provides the same 38-input
packing and action integration without an Isaac Lab runtime.

## Training

1. Run `./train_teacher_quality.sh` to adapt the archived teacher actor to the shared quality reward.
   Its critic, optimizer, iteration, and action standard deviation start fresh; the original teacher checkpoint
   remains untouched. Evaluate saved teacher checkpoints with
   `./evaluate_student.sh /path/to/quality-teacher/model_<iteration>.pt quality_teacher`.
   Approve a teacher only after checking survival, physical step length, touchdown force, heading, and cadence
   across 0.5, 1.0, 1.5, and 2.0 Hz. The teacher-quality task uses the same 45-input policy contract as the
   original teacher. Check alternating touchdown intervals at 0.5 Hz as well as average cadence: a quick pair
   of apparent contacts followed by a long pause can be caused by contact bounce. Confirm detected touchdown
   events against raw contact states and foot motion before interpreting them as uneven steps.
   Use `./evaluate_student.sh /path/to/model_<iteration>.pt quality_teacher 1000 64` for a 64-environment
   frequency sweep. The quality touchdown detector ignores contacts within 0.15 s of the prior accepted
   landing, which removes the short contact bounce seen in the 0.5 Hz trace.
2. Run `./train_student.sh /path/to/approved-quality-teacher/model_<iteration>.pt` to distill the new teacher.
   The script checks the teacher run metadata, so it rejects archived old-objective checkpoints.
   The rollout starts with teacher actions and
   reduces the teacher action coefficient to 70% over 600 updates, while the student learns to match deterministic teacher
   actions. Rollout actions interpolate between teacher and student outputs to avoid abrupt switches.
   Resuming student distillation disables the initial frequency ramp so the restored student keeps training
   across the full 0.5–2.0 Hz command range.
   The frequency ceiling grows from 1.0 to 2.0 Hz over roughly the first 100 updates so the
   student then trains on the full 0.5–2.0 Hz range. The teacher is frozen; only the student is optimized.
   During training, 25% of sampled commands are exactly 0.5 Hz and 25% are exactly the current upper
   frequency limit. This rehearses both evaluation endpoints while preserving continuous commands between them.
   The new student starts with 0.03 action standard deviation and saves every 25 updates; select the checkpoint
   by independent playback because teacher-guided rollout reward can hide student-only falls.
   `./run_student_quality.sh 0.5` plays the latest saved quality-student checkpoint in Newton at 0.5 Hz;
   `./run_student_quality.sh range` displays 0.5, 1.0, 1.5, and 2.0 Hz side by side. Pass a checkpoint as the
   second argument to compare earlier iterations.
3. Evaluate the distilled checkpoint without teacher action mixing:

   ```bash
   ./evaluate_student.sh /path/to/quality-student/model_<iteration>.pt quality_student
   ```

4. Run `./finetune_student.sh /path/to/quality-student/model_<iteration>.pt` only if the independent
   student still needs a small nominal correction.
   Nominal PPO initializes only the student actor; the critic, optimizer, and iteration start fresh. The actor
   sees the 38 deployable observations, while the training-only critic may use the teacher's 45 observations.
   The action standard deviation is fixed at 0.015 and PPO uses a 1e-5 learning rate with two learning epochs
   to limit drift from the distilled gait. No physics randomization is applied in this recovery stage. A 50-update
   pilot from model 374 reduced low-frequency stability and cadence, so do not promote a nominal PPO checkpoint
   without independent evaluation.
5. Robust PPO can start from an approved quality student or nominal PPO checkpoint:
   `./robust_student.sh /path/to/approved/model_<iteration>.pt`. This uses conservative PhysX startup
   variation: foot/body friction 0.8–1.2, base mass ±5%, joint stiffness ±10%, damping ±20%, and small reset
   offsets. Short planar forces act on the base for 0.16 s at a mean interval of 8 s. Their mass-scaled impulse
   starts at an equivalent 0.05 m/s velocity change and ramps to 0.2 m/s over 10,000 environment steps. PPO
   projects the actor back inside a 2% per-parameter relative RMS distance from its starting checkpoint after
   each update; the critic remains unconstrained. The default run is 100 updates so it can be evaluated before
   continuing. `play_mode` removes randomization for a nominal check; pass `disturbed` as the fifth argument
   to `evaluate_student.sh` to evaluate the full-strength force pulses and startup variation. Resuming robust
   PPO preserves the force-ramp position from the checkpoint's iteration and rollout length. In the first
   100-update force-pulse run, model 25 improved disturbed mean episode length from 793 to 868 steps across
   64 environments, but lowered nominal mean episode length from 969 to 919 and increased nominal frequency
   error. Model 99 regressed further. Keep the archived student as the general-purpose baseline; see
   `output/student_force_anchored_stage1_assessment.md` for the checkpoint comparison.
   Use `./run_student_robust.sh range /path/to/robust/model_<iteration>.pt disturbed` for Newton playback
   with a visual stress test. It samples pulses with a mean interval of 1 s, applies each for 0.3 s, and uses
   a mass-scaled equivalent velocity change of 0.25–0.5 m/s; this exceeds the training range of 0–0.2 m/s.
   Training and quantitative evaluation keep their original 8 s mean interval and 0.16 s duration. A red arrow
   above the base shows direction and relative impulse magnitude, with a red sphere marking the active pulse.
   Both disappear when the pulse ends. Omit `disturbed` for nominal playback.

### 1.5 Hz force specialization

`./train_student_15hz.sh` starts from robust model 25 unless a checkpoint or `--resume-latest` is supplied.
It fixes the command at 1.5 Hz and uses random force pulses with a 1 s mean start-to-start interval and 0.3 s
duration. The default impulse range is 0.05–0.25 m/s, overridable with `ZBOT_FORCE_MIN_DELTA_V` and
`ZBOT_FORCE_MAX_DELTA_V`. The default run is 250 additional updates. Calling the script again without an
argument restarts from model 25; `--resume-latest` continues the most recent specialization checkpoint.
It does not change the general-purpose checkpoint in `bests`.

Before training, 64-environment independent playback of the source model gave 976.6 mean steps with 3 falls and
0.072 Hz mean absolute frequency error in the nominal 1.5 Hz task. Under the current 0.05–0.25 m/s training
force distribution, it gave 654.2 mean steps with 38 falls and 0.140 Hz frequency error. Compare each saved
specialist checkpoint against both baselines:

```bash
./train_student_15hz.sh
./evaluate_student_15hz.sh /path/to/specialist/model_<iteration>.pt nominal
./evaluate_student_15hz.sh /path/to/specialist/model_<iteration>.pt training
```

For Newton playback of the newest specialist at 1.5 Hz, run `./run_student_15hz.sh`. It reads the saved
training force range from the checkpoint's `params/env.yaml`, so a manually changed disturbance range is
reflected in playback. Use `./run_student_15hz.sh nominal` to view the same checkpoint without perturbations.

Promote only if disturbed survival improves while nominal cadence and survival remain acceptable. The specialist
is intended for a fixed 1.5 Hz command; test the full frequency range separately before using it as a general
student replacement.

### Student quality fine-tuning

The quality teacher, quality distillation environment, and nominal/robust student PPO stages share one reward
configuration. The student's actor still receives only the 38 deployable observations.
They additionally penalize foot-height difference above 4 cm during single support, touchdown force above
130 N after a real swing, joint torque, and heading/yaw error. Step-length reward saturates at 0.35 m; forward
speed is evaluated but has no separate reward. A positive touchdown-frequency reward compares alternating
landing rate with **twice** the commanded full-cycle frequency. The command term logs measured frequency in
full-cycle Hz, so its per-frequency error metrics use the same units as the command. The 0.5–2.0 Hz command
range and 25% sampling of each endpoint remain active throughout both stages.

From the project root, use the approved checkpoints in sequence:

```bash
./train_teacher_quality.sh                    # actor-only warm start from archived old teacher
./evaluate_student.sh /path/to/quality-teacher/model_25.pt quality_teacher
./train_student.sh /path/to/approved-quality-teacher/model_<iteration>.pt
./evaluate_student.sh /path/to/quality-student/model_50.pt quality_student
./finetune_student.sh /path/to/approved-quality-student/model_<iteration>.pt  # optional, evaluate before use
./evaluate_student.sh /path/to/nominal/model_50.pt nominal
./robust_student.sh /path/to/approved-quality-student-or-nominal/model_<iteration>.pt
./evaluate_student.sh /path/to/robust/model_50.pt robust 1000 64 nominal
./evaluate_student.sh /path/to/robust/model_50.pt robust 1000 64 disturbed
# If interrupted, resume the same stage from its own checkpoint:
./train_teacher_quality.sh --resume /path/to/quality-teacher/model_<iteration>.pt
./train_student.sh --resume /path/to/quality-student/model_<iteration>.pt
./finetune_student.sh --resume /path/to/nominal/model_<iteration>.pt
./robust_student.sh --resume /path/to/robust/model_<iteration>.pt
```

The scripts accept environment count and update count as optional arguments; see their usage lines. Evaluate
at 0.5, 1.0, 1.5, and 2.0 Hz before selecting a checkpoint. The evaluator reports episode length, forward
distance, 90th percentile single-support foot-height difference and landing force, mean absolute heading
error, 90th percentile physical step length, mean absolute cadence error, and cadence event count. Nominal PPO
saves every 25 updates so checkpoints can be selected by these rollout metrics. Reject a checkpoint if improving impact or
heading causes a frequency endpoint to lose stable walking or cadence tracking.

## Hardware alignment before deployment

The current input contract assumes that the IMU axes are aligned with the simulated `base`, that torque
feedback is reported in N·m with the same joint sign/order as `joint1`…`joint6`, and that the controller
runs every 20 ms. Confirm the real IMU-to-base rotation, MIT torque scaling/sign, actuator response,
and loop timing against hardware measurements. The IMU yaw estimate needs a repeatable initial zero and
its drift over a trial must be measured; gravity and gyro alone cannot provide drift-free absolute yaw.
Recalibrate the randomization ranges from those measurements.
The output is six raw joint-velocity actions: apply `tanh`, multiply by `2π rad/s`, integrate for 20 ms,
and clamp the accumulated position offset to ±`π/2` from the default pose. Reset GRU state, action memory,
and phase together at the start of each trial. Do not feed torque commands or absolute position targets
directly to the student in place of the specified feedback values.

This workflow follows the Isaac Lab 3.0 `train-rl-agents`, `domain-randomization-events`, and
`use-sensors-actuators` skills and the Isaac Lab sim-to-real deployment guidance. Validate deterministic
simulation playback and the real sensor/actuator contract before allowing physical walking.

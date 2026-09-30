# Student fine-tuning collection performance

Measured on 2026-09-27 with a GeForce RTX 5090, `physics=isaacsim_physx`, no visualizer, and the 38-input student. All measurements below are playback or random-action benchmarks; no training was running during these tests.

## Findings

The stopped 4096-environment PPO runs logged about 2.5–3.3 seconds of collection and 0.15–0.18 seconds of learning per update. Collection accounted for 94–96% of iteration time.

The command-frequency metric used dynamic CUDA boolean indexing for four bins on every environment step and 16 individual `.item()` transfers during reset. It now uses fixed-shape `scatter_add_` and one batched host transfer. Touchdown state updates use fixed-shape writes. In an isolated 4096-environment metric benchmark, the frequency-metric function fell from 1.67 to 0.06 ms per call. A single end-to-end 4096-environment checkpoint-playback comparison rose from about 67.1k to 69.8k steps/s. This is an observed small improvement, not a guaranteed training-speed gain.

| Environments | Random-action runtime | Student checkpoint playback | Playback step time | Scene creation |
| ---: | ---: | ---: | ---: | ---: |
| 2048 | 37.6k steps/s | — | — | 15 s |
| 4096 | 68.1k steps/s | 69.8k steps/s | 58.7 ms | 27 s |
| 8192 | 104.8k steps/s | 103.7k steps/s | 79.0 ms | 52 s |
| 16384 | 145.0k steps/s | 139.2k steps/s | 117.7 ms | 105 s |

The playback run used `model_100.pt` from `2026-09-27_08-47-13_student_quality_nominal`. The 16384-environment run measured 120 steps after 30 warmup steps; smaller configurations used 200 after 50. Rates are aggregate environment steps per second. More environments increase total throughput while making each individual environment step and each fixed-horizon PPO update take longer.

A synchronized 4096-environment playback diagnostic measured 32.8 ms inside simulation and 25.5 ms outside it per step. This diagnostic changes scheduling and must not be interpreted as normal throughput. The remainder includes actions, rewards, observations, resets, and synchronization; it is not entirely removable overhead.

## Choice for the next training run

Use 8192 environments as the first throughput experiment with the unchanged 48-step PPO horizon and 200 Hz PhysX / 50 Hz policy timing. It offers about 49% more checkpoint-playback throughput than 4096 with moderate startup and memory cost. Try 16384 only if the larger PPO batch fits memory and its learning and frequency metrics remain acceptable. The training scripts already accept environment count as a positional argument.

Keep the 200 Hz physics step. A 100 Hz test (2 physics steps per policy action) raised random-action throughput to 100.5k steps/s at 4096 environments, but independent-student playback fell to 500 steps at 1.5 Hz and 200 steps at 2.0 Hz. A 150 Hz test also reduced the 2.0 Hz episode to 250 steps. Those changes would invalidate the contact and frequency behavior targeted by this fine-tune.

Disabling command debug visualization changed 4096-environment playback from about 70.4k to 70.6k steps/s, within measurement variation. The task already runs headless. A later physics-focused experiment could simplify the robot's 12 convex-hull collision meshes, but must recheck foot contacts, impact forces, and frequency tracking.

Benchmark JSON files are under `output/performance_benchmark/`. Re-run a fixed checkpoint and compare `runtime.environment_step_timing.environment_step_fps` across configurations; compare PPO runs by total frames and independent-student quality rather than update count alone.

#!/usr/bin/env python3
"""Monitor a ZBot TensorBoard run and gracefully stop failed training trends."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from datetime import datetime
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS = {
    "iteration": "Train/mean_reward",
    "reward": "Train/mean_reward",
    "episode_length": "Train/mean_episode_length",
    "stage": "Curriculum/walking_stages/stage",
    "ready": "Curriculum/walking_stages/promotion_ready",
    "blend": "Curriculum/walking_stages/stage_two_blend",
    "survival": "Curriculum/walking_stages/survival_ratio_ema",
    "cadence": "Curriculum/walking_stages/alternation_rate_ema",
    "frequency_reward": "Episode_Reward/stage_one_step_frequency",
    "frequency_penalty": "Episode_Reward/step_frequency_error_l2",
    "touchdown_reward": "Episode_Reward/alternating_touchdown",
    "forward_reward": "Episode_Reward/body_forward_speed",
    "step_length_reward": "Episode_Reward/step_length",
    "step_asymmetry_penalty": "Episode_Reward/step_length_asymmetry",
    "action_std": "Policy/mean_std",
    "learning_rate": "Loss/learning_rate",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True, help="Training process ID.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True, help="Suffix used by the target run.")
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true", help="Record stop decisions without signaling the process.")
    return parser.parse_args()


def _process_is_training(pid: int) -> bool:
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return False
    return "train_compact.py" in command or "isaaclab train" in command


def _find_run(run_root: Path, run_name: str) -> Path | None:
    candidates = [path for path in run_root.glob(f"*_{run_name}") if path.is_dir()]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _load_series(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    event_files = sorted(run_dir.glob("events.out.tfevents.*"), key=lambda path: path.stat().st_mtime)
    if not event_files:
        return {}
    accumulator = EventAccumulator(str(event_files[-1]), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))
    return {
        name: [(event.step, event.value) for event in accumulator.Scalars(tag)]
        for name, tag in TAGS.items()
        if tag in available
    }


def _tail_mean(series: list[tuple[int, float]], count: int = 20) -> float:
    tail = series[-count:]
    return sum(value for _, value in tail) / len(tail) if tail else math.nan


def _window_delta(series: list[tuple[int, float]], step_window: int = 500, edge_count: int = 20) -> float:
    """Return the change between smoothed window edges, or NaN with insufficient history."""
    if not series:
        return math.nan
    latest_step = series[-1][0]
    window = [value for step, value in series if step >= latest_step - step_window]
    if len(window) < 2 * edge_count:
        return math.nan
    return sum(window[-edge_count:]) / edge_count - sum(window[:edge_count]) / edge_count


def evaluate_trend(series: dict[str, list[tuple[int, float]]]) -> tuple[dict[str, float], str]:
    """Return the latest trend snapshot and a continue/stop decision."""
    snapshot = {name: values[-1][1] for name, values in series.items() if values}
    iteration = int(series.get("iteration", [(0, 0.0)])[-1][0])
    snapshot["iteration"] = float(iteration)
    core_values = [value for name, value in snapshot.items() if name != "iteration"]
    if any(not math.isfinite(value) for value in core_values):
        return snapshot, "stop_non_finite"

    cadence = _tail_mean(series.get("cadence", []), 50)
    survival = _tail_mean(series.get("survival", []), 50)
    frequency_reward = _tail_mean(series.get("frequency_reward", []), 50)
    action_std = _tail_mean(series.get("action_std", []), 20)
    forward_reward = _tail_mean(series.get("forward_reward", []), 100)
    step_length_reward = _tail_mean(series.get("step_length_reward", []), 100)
    reward_delta = _window_delta(series.get("reward", []))
    forward_delta = _window_delta(series.get("forward_reward", []))
    ready = snapshot.get("ready", 0.0) >= 0.5
    stage = snapshot.get("stage", 1.0)

    if iteration >= 600 and stage < 2.0 and not ready and survival >= 0.8 and cadence < 0.2:
        return snapshot, "stop_no_gait"
    if iteration >= 300 and stage < 2.0 and not ready and survival >= 0.8 and cadence < 0.05 and action_std < 0.2:
        return snapshot, "stop_no_gait_collapse"
    if iteration >= 600 and stage < 2.0 and not ready and cadence > 4.0 and frequency_reward < 0.01:
        return snapshot, "stop_cadence_exploit"
    if iteration >= 1000 and stage < 2.0 and not ready and action_std < 0.04:
        return snapshot, "stop_exploration_collapse"
    if iteration >= 700 and stage >= 2.0 and snapshot.get("blend", 0.0) >= 0.99 and survival < 0.4:
        return snapshot, "stop_unstable_stage_two"
    if (
        iteration >= 1000
        and stage >= 2.0
        and snapshot.get("blend", 0.0) >= 0.99
        and forward_reward < 0.03
        and step_length_reward < 0.02
    ):
        return snapshot, "stop_no_forward_progress"
    if iteration >= 1000 and action_std < 0.04:
        return snapshot, "stop_exploration_collapse"
    if iteration >= 600 and stage >= 2.0 and snapshot.get("blend", 0.0) >= 0.99 and cadence > 4.0:
        return snapshot, "stop_stage_two_cadence_exploit"
    if (
        iteration >= 2500
        and stage >= 2.0
        and 1.0 <= cadence <= 2.0
        and survival >= 0.8
        and reward_delta < 0.5
        and forward_delta < 0.03
    ):
        return snapshot, "stop_converged_plateau"
    return snapshot, "continue"


def main() -> None:
    args = _parse_args()
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    with args.log_file.open("a", encoding="utf-8", buffering=1) as log:
        while _process_is_training(args.pid):
            run_dir = _find_run(args.run_root, args.run_name)
            if run_dir is None:
                record = {"timestamp": datetime.now().isoformat(), "decision": "waiting_for_run"}
            else:
                try:
                    series = _load_series(run_dir)
                    snapshot, decision = evaluate_trend(series) if series else ({}, "waiting_for_metrics")
                    record = {
                        "timestamp": datetime.now().isoformat(),
                        "run_dir": str(run_dir),
                        "decision": decision,
                        **snapshot,
                    }
                except Exception as error:  # keep monitoring after a partially-written event record
                    record = {
                        "timestamp": datetime.now().isoformat(),
                        "run_dir": str(run_dir),
                        "decision": "monitor_read_error",
                        "error": repr(error),
                    }
            line = json.dumps(record, ensure_ascii=False, sort_keys=True)
            print(line, flush=True)
            log.write(line + "\n")
            if str(record["decision"]).startswith("stop_"):
                if not args.dry_run and _process_is_training(args.pid):
                    os.kill(args.pid, signal.SIGINT)
                return
            time.sleep(args.interval)

        record = {"timestamp": datetime.now().isoformat(), "decision": "training_exited"}
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        print(line, flush=True)
        log.write(line + "\n")


if __name__ == "__main__":
    main()

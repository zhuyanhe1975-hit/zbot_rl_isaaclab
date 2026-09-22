from scripts.monitor_training import evaluate_trend


def _series(iteration: int, *, cadence: float, survival: float, ready: float = 0.0, action_std: float = 0.1):
    return {
        "iteration": [(iteration, 3.0)],
        "stage": [(iteration, 1.0)],
        "ready": [(iteration, ready)],
        "cadence": [(iteration, cadence)],
        "survival": [(iteration, survival)],
        "frequency_reward": [(iteration, 0.0)],
        "action_std": [(iteration, action_std)],
    }


def test_monitor_stops_stable_policy_that_never_steps():
    _, decision = evaluate_trend(_series(600, cadence=0.0, survival=0.95))
    assert decision == "stop_no_gait"


def test_monitor_stops_early_when_standing_policy_has_collapsed():
    _, decision = evaluate_trend(_series(300, cadence=0.0, survival=0.95, action_std=0.1))
    assert decision == "stop_no_gait_collapse"


def test_monitor_stops_high_frequency_reward_exploit():
    _, decision = evaluate_trend(_series(600, cadence=8.0, survival=0.95))
    assert decision == "stop_cadence_exploit"


def test_monitor_keeps_qualified_gait_running():
    _, decision = evaluate_trend(_series(600, cadence=1.5, survival=0.95, ready=1.0))
    assert decision == "continue"


def test_monitor_stops_stage_two_without_forward_progress():
    series = _series(1000, cadence=1.5, survival=0.95, ready=1.0)
    series["stage"] = [(1000, 2.0)]
    series["blend"] = [(1000, 1.0)]
    series["forward_reward"] = [(1000, 0.0)]
    series["step_length_reward"] = [(1000, 0.0)]

    _, decision = evaluate_trend(series)

    assert decision == "stop_no_forward_progress"


def test_monitor_keeps_productive_stage_two_running():
    series = _series(1000, cadence=2.5, survival=0.9, ready=1.0)
    series["stage"] = [(1000, 2.0)]
    series["blend"] = [(1000, 1.0)]
    series["forward_reward"] = [(1000, 0.2)]
    series["step_length_reward"] = [(1000, 0.1)]

    _, decision = evaluate_trend(series)

    assert decision == "continue"


def test_monitor_stops_stage_two_cadence_exploit():
    series = _series(600, cadence=5.0, survival=0.9, ready=1.0)
    series["stage"] = [(600, 2.0)]
    series["blend"] = [(600, 1.0)]

    _, decision = evaluate_trend(series)

    assert decision == "stop_stage_two_cadence_exploit"


def test_monitor_stops_converged_stage_two_plateau():
    series = _series(2500, cadence=1.8, survival=0.9, ready=1.0)
    series["stage"] = [(step, 2.0) for step in range(1900, 2501)]
    series["blend"] = [(step, 1.0) for step in range(1900, 2501)]
    series["reward"] = [(step, 40.0) for step in range(1900, 2501)]
    series["forward_reward"] = [(step, 1.0) for step in range(1900, 2501)]
    series["step_length_reward"] = [(step, 0.5) for step in range(1900, 2501)]

    _, decision = evaluate_trend(series)

    assert decision == "stop_converged_plateau"

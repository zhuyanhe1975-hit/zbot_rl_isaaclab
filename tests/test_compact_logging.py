from types import SimpleNamespace

from zbot_rl_isaaclab.compact_logging import _nonzero_reward_weights, format_compact_training_log


def test_compact_training_log_groups_the_key_metrics():
    output = format_compact_training_log(
        iteration=1642,
        total_iterations=3000,
        total_steps=16_151_342,
        steps_per_second=132_713,
        iteration_time=0.74,
        elapsed_seconds=1160.0,
        eta_seconds=958.0,
        losses={"value": 0.0, "surrogate": 0.0032, "entropy": -4.298},
        learning_rate=1.0e-3,
        action_std=0.12,
        mean_reward=3.4,
        mean_episode_length=990.1,
        metrics={
            "Curriculum/walking_stages/stage": 1.0,
            "Curriculum/walking_stages/promotion_ready": 0.0,
            "Curriculum/walking_stages/stage_two_blend": 0.0,
            "Curriculum/walking_stages/survival_ratio_ema": 0.9857,
            "Curriculum/walking_stages/alternation_rate_ema": 0.0,
            "Episode_Reward/alive": 0.1979,
            "Episode_Reward/alternating_touchdown": 0.0,
            "Episode_Reward/stage_one_step_frequency": 1.2,
            "Episode_Reward/joint_deviation": -0.0132,
            "Episode_Reward/action_rate_l2": -0.0033,
            "Episode_Reward/joint_torques_l2": -8.4e-5,
            "Episode_Reward/disabled": 99.0,
            "Episode_Termination/time_out": 0.9761,
            "Episode_Termination/base_height": 0.0239,
        },
        reward_weights={
            "alive": 0.2,
            "alternating_touchdown": 1.0e-6,
            "stage_one_step_frequency": 50.0,
            "joint_deviation": -0.1,
            "action_rate_l2": -0.01,
            "joint_torques_l2": -1.0e-6,
            "disabled": 0.0,
        },
    )

    assert "[Performance]" in output
    assert "[Optimization]" in output
    assert "[Episode]" in output
    assert "[Curriculum] stage=1  ready=no" in output
    assert "[Rewards]" in output
    assert "alternating_touchdown=+0.000 (scale=1e-06)" in output
    assert "stage_one_step_frequency=+1.200 (scale=50)" in output
    assert "joint_deviation=-0.013 (scale=-0.1)" in output
    assert "joint_torques_l2=-8.400e-05 (scale=-1e-06)" in output
    assert "disabled" not in output
    assert "[Termination] timeout=97.6%  base_height=2.4%" in output


def test_base_task_log_only_shows_balance_transfer_metrics():
    output = format_compact_training_log(
        iteration=85,
        total_iterations=1500,
        total_steps=1_056_768,
        steps_per_second=8_844,
        iteration_time=1.39,
        elapsed_seconds=158.0,
        eta_seconds=2603.0,
        losses={"value": 0.4488, "surrogate": -0.0044, "entropy": 6.8802},
        learning_rate=5.0e-4,
        action_std=0.762,
        mean_reward=69.88,
        mean_episode_length=954.8,
        metrics={
            "Episode_Reward/alive": 0.496,
            "Episode_Reward/support_distance": 2.5,
            "Episode_Reward/support_force": 1.4,
            "Episode_Reward/joint_deviation_l1": -0.03,
            "Episode_Reward/vertical_velocity_l2": -0.01,
            "Episode_Reward/angular_velocity_l2": -0.015,
            "Episode_Reward/action_rate_l2": -0.1,
            "Episode_Reward/joint_torques_l2": -0.001,
            "Episode_Termination/time_out": 0.869,
            "Episode_Termination/base_height": 0.10,
            "Episode_Termination/illegal_contact": 0.02,
            "Episode_Termination/feet_collision": 0.011,
        },
        reward_weights={
            "alive": 2.0,
            "support_distance": 2.0,
            "support_force": 2.0,
            "termination_penalty": -20.0,
            "joint_deviation_l1": -0.2,
            "vertical_velocity_l2": -500.0,
            "angular_velocity_l2": -0.5,
            "action_rate_l2": -0.01,
            "joint_torques_l2": 0.0,
        },
        task_profile="base",
    )

    assert "[Episode] reward=69.88  length=954.8" in output
    assert "support_distance=+2.500 (scale=2)" in output
    assert "support_force=+1.400 (scale=2)" in output
    assert "joint_deviation_l1=-0.030 (scale=-0.2)" in output
    assert "joint_torques_l2" not in output
    assert "feet_collision=1.1%" in output
    assert "Curriculum" not in output
    assert "touchdown" not in output
    assert "double_support" not in output
    assert "frequency" not in output
    assert "forward" not in output
    assert "step=" not in output
    assert "survival" not in output


def test_direct_walking_log_does_not_claim_an_active_curriculum():
    output = format_compact_training_log(
        iteration=10,
        total_iterations=3000,
        total_steps=1000,
        steps_per_second=100,
        iteration_time=1.0,
        elapsed_seconds=10.0,
        eta_seconds=100.0,
        losses={"value": 0.1},
        learning_rate=5.0e-4,
        action_std=0.7,
        mean_reward=1.0,
        mean_episode_length=500.0,
        metrics={
            "Episode_Reward/step_length": 0.0,
            "Episode_Reward/disabled": 5.0,
            "Metrics/step_frequency/mean_target_frequency": 1.25,
            "Metrics/step_frequency/measured_sum_0p5": 12.4,
            "Metrics/step_frequency/signed_error_sum_0p5": 2.4,
            "Metrics/step_frequency/absolute_error_sum_0p5": 2.4,
            "Metrics/step_frequency/samples_0p5": 20.0,
            "Metrics/step_frequency/measured_sum_2": 40.04,
            "Metrics/step_frequency/signed_error_sum_2": -1.98,
            "Metrics/step_frequency/absolute_error_sum_2": 3.96,
            "Metrics/step_frequency/samples_2": 22.0,
            "Episode_Termination/time_out": 0.8,
        },
        reward_weights={"step_length": 50.0, "disabled": 0.0},
    )

    assert "[Mode] direct walking" in output
    assert "actor prior" not in output
    assert "[Curriculum]" not in output
    assert "survival=80.0%" in output
    assert "step_length=+0.000 (scale=50)" in output
    assert "disabled" not in output
    assert "[Command] target_frequency=1.250Hz" in output
    assert "[Frequency] 0.5->0.62Hz (bias=0.12, mae=0.12, n=20)" in output
    assert "2->1.82Hz (bias=0.09, mae=0.18, n=22)" in output


def test_nonzero_reward_weights_supports_object_and_mapping_configs():
    object_cfg = SimpleNamespace(
        rewards=SimpleNamespace(
            active=SimpleNamespace(weight=2.0),
            disabled=SimpleNamespace(weight=0.0),
            missing=None,
        )
    )
    mapping_cfg = {
        "rewards": {
            "penalty": {"weight": -0.25},
            "disabled": {"weight": 0.0},
        }
    }

    assert _nonzero_reward_weights(object_cfg) == {"active": 2.0}
    assert _nonzero_reward_weights(mapping_cfg) == {"penalty": -0.25}

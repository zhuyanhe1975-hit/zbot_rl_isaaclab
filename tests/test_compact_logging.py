from zbot_rl_isaaclab.compact_logging import format_compact_training_log


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
            "Episode_Reward/joint_deviation": -0.0132,
            "Episode_Reward/action_rate_l2": -0.0033,
            "Episode_Termination/time_out": 0.9761,
            "Episode_Termination/base_height": 0.0239,
        },
    )

    assert "[Performance]" in output
    assert "[Optimization]" in output
    assert "[Episode]" in output
    assert "[Curriculum] stage=1  ready=no" in output
    assert "[Rewards]" in output
    assert "[Penalties] balance=+0.000  control=-0.017" in output
    assert "[Termination] timeout=97.6%  base_height=2.4%" in output
    assert len(output.splitlines()) == 9


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
        task_profile="base",
    )

    assert "[Episode] reward=69.88  length=954.8" in output
    assert "support_distance=+2.500  support_force=+1.400" in output
    assert "[Penalties] joint_pose=-0.030  motion=-0.025  control=-0.100" in output
    assert "feet_collision=1.1%" in output
    assert "Curriculum" not in output
    assert "touchdown" not in output
    assert "double_support" not in output
    assert "frequency" not in output
    assert "forward" not in output
    assert "step=" not in output
    assert "survival" not in output
    assert len(output.splitlines()) == 8

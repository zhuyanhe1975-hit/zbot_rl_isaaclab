import isaaclab.envs.mdp as base_mdp

from zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof_frequency_balance.env_cfg import (
    FREQUENCY_BALANCE_OBSERVATION_DIM,
    FrequencyBalanceEnvCfg,
)
from zbot_rl_isaaclab.tasks.velocity.mdp.rewards import (
    commanded_support_joint_force_contrast,
    ideal_support_foot_distance_reward,
)


def test_frequency_balance_task_conditions_dense_weight_transfer_on_frequency():
    cfg = FrequencyBalanceEnvCfg()

    assert FREQUENCY_BALANCE_OBSERVATION_DIM == 42
    assert cfg.commands.weight_shift.minimum_frequency == 0.5
    assert cfg.commands.weight_shift.maximum_frequency == 2.0
    assert cfg.observations.policy.ideal_support_foot.func is base_mdp.generated_commands
    assert cfg.rewards.support_distance.func is ideal_support_foot_distance_reward
    assert cfg.rewards.support_distance.params["command_index"] == 1
    assert cfg.rewards.support_force.func is commanded_support_joint_force_contrast
    assert cfg.rewards.support_force.params["command_index"] == 1
    assert not hasattr(cfg.rewards, "step_length")
    assert not hasattr(cfg.rewards, "alternating_touchdown")


def test_frequency_balance_play_mode_orders_and_holds_frequency_commands():
    cfg = FrequencyBalanceEnvCfg()
    cfg.play_mode()

    assert cfg.commands.weight_shift.ordered
    assert cfg.commands.weight_shift.resampling_time_range == (1.0e9, 1.0e9)

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import torch

from zbot_rl_isaaclab.tasks.velocity.mdp.rewards import StepLengthReward, update_step_length_state


def test_oversized_touchdown_keeps_raw_length_but_caps_reward():
    reward_term = StepLengthReward.__new__(StepLengthReward)
    reward_term._sensor_foot_ids = [0, 1]
    reward_term._asset_foot_ids = [0, 1]
    reward_term._contact_sensor = SimpleNamespace(
        data=SimpleNamespace(
            last_air_time=SimpleNamespace(torch=torch.tensor([[0.1, 0.0]])),
            current_air_time=SimpleNamespace(torch=torch.zeros(1, 2)),
            net_normal_forces_w_history=SimpleNamespace(torch=torch.tensor([[[[0.0, 0.0, 20.0], [0.0, 0.0, 0.0]]]])),
        ),
        compute_first_contact=lambda _: SimpleNamespace(torch=torch.tensor([[True, False]])),
    )
    reward_term._asset = SimpleNamespace(
        data=SimpleNamespace(
            body_pos_w=SimpleNamespace(torch=torch.tensor([[[0.55, 0.0, 0.0], [0.0, 0.0, 0.0]]])),
            root_quat_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, 0.0, 1.0]])),
        )
    )
    reward_term._previous_touchdown_pos_w = torch.zeros(1, 2, 3)
    reward_term._last_step_length = torch.zeros(1, 2)
    reward_term._has_step_length = torch.zeros(1, 2, dtype=torch.bool)
    reward_term._was_behind = torch.tensor([[True, False]])

    value = reward_term(
        SimpleNamespace(step_dt=0.02),
        sensor_cfg=None,
        asset_cfg=None,
        minimum_air_time=0.05,
        force_threshold=10.0,
        crossing_margin=0.0,
        return_symmetry_error=False,
        maximum_rewarded_step_length=0.35,
    )

    torch.testing.assert_close(value, torch.tensor([0.35]))
    torch.testing.assert_close(reward_term._last_step_length, torch.tensor([[0.55, 0.0]]))


def test_step_length_reward_is_unbounded_and_preserves_metric_units():
    """Every positive landing length must pass through without a target or upper cap."""
    forward_step_length = torch.tensor([[0.01, 0.0], [0.0, 0.50]])
    valid_touchdown = torch.tensor([[True, False], [False, True]])

    reward, _, _, _ = update_step_length_state(
        forward_step_length,
        valid_touchdown,
        last_step_length=torch.zeros(2, 2),
        has_step_length=torch.zeros(2, 2, dtype=torch.bool),
    )

    torch.testing.assert_close(reward, torch.tensor([0.01, 0.50]))


def test_airborne_motion_without_touchdown_has_zero_step_length():
    """A foot displacement cannot earn step-length reward before a valid landing."""
    reward, symmetry_error, updated_length, updated_seen = update_step_length_state(
        forward_step_length=torch.tensor([[10.0, 8.0]]),
        valid_touchdown=torch.zeros(1, 2, dtype=torch.bool),
        last_step_length=torch.tensor([[0.20, 0.25]]),
        has_step_length=torch.tensor([[True, True]]),
    )

    torch.testing.assert_close(reward, torch.zeros(1))
    torch.testing.assert_close(symmetry_error, torch.zeros(1))
    torch.testing.assert_close(updated_length, torch.tensor([[0.20, 0.25]]))
    assert torch.equal(updated_seen, torch.tensor([[True, True]]))


def test_step_length_symmetry_compares_latest_left_and_right_steps():
    """Symmetry error must be emitted only after both feet have measured steps."""
    forward_step_length = torch.tensor([[0.30, 0.0], [0.0, 0.50]])
    valid_touchdown = torch.tensor([[True, False], [False, True]])
    last_step_length = torch.tensor([[0.0, 0.0], [0.20, 0.0]])
    has_step_length = torch.tensor([[False, False], [True, False]])

    _, symmetry_error, updated_length, updated_seen = update_step_length_state(
        forward_step_length,
        valid_touchdown,
        last_step_length,
        has_step_length,
    )

    torch.testing.assert_close(symmetry_error, torch.tensor([0.0, 0.30]))
    torch.testing.assert_close(updated_length, torch.tensor([[0.30, 0.0], [0.20, 0.50]]))
    assert torch.equal(updated_seen, torch.tensor([[True, False], [True, True]]))

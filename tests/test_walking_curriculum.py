# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from zbot_rl_isaaclab.tasks.velocity.mdp.curriculums import (
    compute_stepping_promotion_metrics,
    meets_stepping_promotion_gate,
    stage_two_blend,
)


def test_stepping_promotion_metrics_ignore_empty_episodes():
    """Promotion metrics must use only environments that completed non-empty episodes."""
    survival_ratio, alternation_rate = compute_stepping_promotion_metrics(
        episode_length_steps=torch.tensor([0, 600, 1000]),
        maximum_episode_steps=1000,
        alternating_steps=torch.tensor([100.0, 6.0, 20.0]),
        step_dt=0.02,
    )

    torch.testing.assert_close(survival_ratio, torch.tensor(0.8))
    torch.testing.assert_close(alternation_rate, torch.tensor(0.75))


def test_stage_two_blend_ramps_after_promotion():
    """Stage-two rewards must turn on smoothly after promotion."""
    assert stage_two_blend(100, promotion_step=-1, transition_steps=100) == 0.0
    assert stage_two_blend(100, promotion_step=100, transition_steps=100) == 0.0
    assert stage_two_blend(150, promotion_step=100, transition_steps=100) == 0.5
    assert stage_two_blend(250, promotion_step=100, transition_steps=100) == 1.0


def test_promotion_requires_frequency_inside_one_to_two_hertz():
    """Stable but too-slow or too-fast stepping must remain in stage one."""
    common = {
        "common_step_counter": 5_000,
        "minimum_training_steps": 5_000,
        "survival_ratio": 0.8,
        "survival_ratio_threshold": 0.6,
        "minimum_alternation_frequency": 1.0,
        "maximum_alternation_frequency": 2.0,
    }

    assert not meets_stepping_promotion_gate(alternation_frequency=0.9, **common)
    assert meets_stepping_promotion_gate(alternation_frequency=1.0, **common)
    assert meets_stepping_promotion_gate(alternation_frequency=1.5, **common)
    assert meets_stepping_promotion_gate(alternation_frequency=2.0, **common)
    assert not meets_stepping_promotion_gate(alternation_frequency=2.1, **common)

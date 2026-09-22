# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import torch

from zbot_rl_isaaclab.tasks.velocity.mdp.curriculums import (
    compute_stepping_promotion_metrics,
    frequency_control_progress,
    meets_stepping_performance_gate,
    stability_progress,
    stepping_performance_blend,
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


def test_stage_two_blend_tracks_performance_instead_of_elapsed_steps():
    common = {
        "survival_ratio": 0.8,
        "survival_ratio_threshold": 0.6,
        "minimum_alternation_frequency": 1.0,
        "maximum_alternation_frequency": 2.0,
        "frequency_tolerance": 0.5,
    }

    assert stepping_performance_blend(alternation_frequency=1.5, **common) == 1.0
    assert 0.0 < stepping_performance_blend(alternation_frequency=0.5, **common) < 1.0
    assert stepping_performance_blend(alternation_frequency=0.0, **(common | {"survival_ratio": 0.0})) == 0.0
    assert stepping_performance_blend(alternation_frequency=1.5, **(common | {"survival_ratio": 0.59})) == 0.0


def test_stability_progress_scales_stage_one_shaping_without_elapsed_time():
    assert stability_progress(0.0, 0.6) == 0.0
    assert stability_progress(0.3, 0.6) == 0.5
    assert stability_progress(0.6, 0.6) == 1.0
    assert stability_progress(0.9, 0.6) == 1.0


def test_frequency_control_ramps_only_after_basic_stability():
    assert frequency_control_progress(0.2, 0.3, 0.6) == 0.0
    assert frequency_control_progress(0.3, 0.3, 0.6) == 0.0
    assert math.isclose(frequency_control_progress(0.45, 0.3, 0.6), 0.5)
    assert frequency_control_progress(0.6, 0.3, 0.6) == 1.0


def test_promotion_requires_frequency_inside_one_to_two_hertz():
    """Stable but too-slow or too-fast stepping must remain in stage one."""
    common = {
        "survival_ratio": 0.8,
        "survival_ratio_threshold": 0.6,
        "minimum_alternation_frequency": 1.0,
        "maximum_alternation_frequency": 2.0,
    }

    assert not meets_stepping_performance_gate(alternation_frequency=0.9, **common)
    assert meets_stepping_performance_gate(alternation_frequency=1.0, **common)
    assert meets_stepping_performance_gate(alternation_frequency=1.5, **common)
    assert meets_stepping_performance_gate(alternation_frequency=2.0, **common)
    assert not meets_stepping_performance_gate(alternation_frequency=2.1, **common)


def test_stage_two_blend_falls_when_cadence_leaves_target_band():
    in_band = stepping_performance_blend(0.8, 0.6, 1.5, 1.0, 2.0, 0.5)
    too_fast = stepping_performance_blend(0.8, 0.6, 3.0, 1.0, 2.0, 0.5)

    assert in_band == 1.0
    assert too_fast < in_band

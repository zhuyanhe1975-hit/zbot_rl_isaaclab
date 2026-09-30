from types import SimpleNamespace
from unittest.mock import Mock

import torch

from zbot_rl_isaaclab.tasks.velocity.mdp.events import (
    ExternalForcePulse,
    force_pulse_velocity_limit,
    planar_force_for_velocity_change,
    pulse_start_probability,
)


def test_short_force_pulse_matches_requested_horizontal_impulse():
    mass = torch.tensor([2.0, 4.0])
    delta_velocity = torch.tensor([0.2, 0.1])
    direction = torch.tensor([0.0, torch.pi / 2.0])

    force = planar_force_for_velocity_change(mass, delta_velocity, direction, duration_s=0.2)

    torch.testing.assert_close(force[0], torch.tensor([2.0, 0.0, 0.0]), atol=1e-6, rtol=0)
    torch.testing.assert_close(force[1], torch.tensor([0.0, 2.0, 0.0]), atol=1e-6, rtol=0)


def test_force_pulse_rejects_nonpositive_duration():
    try:
        planar_force_for_velocity_change(torch.ones(1), torch.ones(1), torch.zeros(1), duration_s=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected invalid duration to be rejected")


def test_force_pulse_curriculum_continues_after_resume():
    assert force_pulse_velocity_limit(0.05, 0.2, step=0, offset=0, ramp_steps=10_000) == 0.05
    assert force_pulse_velocity_limit(0.05, 0.2, step=5_000, offset=0, ramp_steps=10_000) == 0.125
    assert force_pulse_velocity_limit(0.05, 0.2, step=0, offset=5_000, ramp_steps=10_000) == 0.125
    assert force_pulse_velocity_limit(0.05, 0.2, step=10_000, offset=5_000, ramp_steps=10_000) == 0.2


def test_force_arrow_shows_direction_and_equivalent_impulse():
    term = ExternalForcePulse.__new__(ExternalForcePulse)
    term._body_ids = [0]
    term._duration_s = 0.2
    term._forces = torch.tensor([[[2.0, 0.0, 0.0]]])
    term._force_visualizer = Mock()
    term._pulse_indicator = Mock()
    term._asset = SimpleNamespace(
        data=SimpleNamespace(
            body_pos_w=SimpleNamespace(torch=torch.tensor([[[1.0, 2.0, 0.3]]])),
            body_quat_w=SimpleNamespace(torch=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])),
            body_mass=SimpleNamespace(torch=torch.tensor([[2.0]])),
        )
    )

    term._visualize()

    positions, vectors = term._force_visualizer.visualize.call_args.args
    torch.testing.assert_close(positions, torch.tensor([[1.0, 2.0, 0.55]]))
    torch.testing.assert_close(vectors, torch.tensor([[0.2, 0.0, 0.0]]))
    torch.testing.assert_close(term._pulse_indicator.visualize.call_args.kwargs["scales"], torch.ones((1, 3)))


def test_random_pulse_probability_accounts_for_visible_pulse_duration():
    assert abs(pulse_start_probability(1.0, 0.4, 0.02) - 1.0 / 30.0) < 1.0e-8

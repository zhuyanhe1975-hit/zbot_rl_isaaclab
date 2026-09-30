from unittest.mock import patch

import torch
from rsl_rl.algorithms import PPO

from zbot_rl_isaaclab.student_ppo import AnchoredPPO


def test_actor_projection_keeps_each_parameter_near_the_student_prior():
    algorithm = AnchoredPPO.__new__(AnchoredPPO)
    algorithm._raw_actor = torch.nn.Linear(3, 2)
    algorithm.max_relative_drift = 0.02
    algorithm.set_reference_actor()
    with torch.no_grad():
        for parameter in algorithm._raw_actor.parameters():
            parameter.add_(0.5)

    before = algorithm._project_actor()

    assert before > 0.02
    for name, parameter in algorithm._raw_actor.named_parameters():
        reference = algorithm._actor_reference[name]
        scale = reference.square().mean().sqrt().clamp_min(0.01)
        drift = (parameter - reference).square().mean().sqrt() / scale
        assert drift <= 0.020001


def test_anchor_is_saved_and_restored_with_ppo_checkpoint():
    algorithm = AnchoredPPO.__new__(AnchoredPPO)
    algorithm._raw_actor = torch.nn.Linear(3, 2)
    algorithm.max_relative_drift = 0.02
    algorithm.set_reference_actor()
    saved_reference = {name: value.clone() for name, value in algorithm._actor_reference.items()}
    with patch.object(PPO, "save", return_value={"actor_state_dict": {}}):
        checkpoint = algorithm.save()

    with torch.no_grad():
        for parameter in algorithm._raw_actor.parameters():
            parameter.add_(0.5)
    with patch.object(PPO, "load", return_value=True):
        assert algorithm.load(checkpoint, None, True)

    for name, value in algorithm._actor_reference.items():
        torch.testing.assert_close(value, saved_reference[name])


def test_old_ppo_checkpoint_becomes_its_own_anchor_on_resume():
    algorithm = AnchoredPPO.__new__(AnchoredPPO)
    algorithm._raw_actor = torch.nn.Linear(3, 2)
    algorithm._actor_reference = None
    with patch.object(PPO, "load", return_value=True):
        assert algorithm.load({"actor_state_dict": {}}, None, True)

    assert set(algorithm._actor_reference) == set(dict(algorithm._raw_actor.named_parameters()))

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace
from typing import cast

import torch

from zbot_rl_isaaclab.pretraining import (
    BALANCE_OBSERVATION_DIM,
    FREQUENCY_BALANCE_OBSERVATION_DIM,
    FREQUENCY_COMMAND_OBSERVATION_DIM,
    PERIODIC_BASE_OBSERVATION_DIM,
    WALKING_OBSERVATION_DIM,
    build_frequency_balance_actor_prior,
    build_periodic_base_continuation_actor_prior,
    build_periodic_base_walking_actor_prior,
    build_walking_actor_prior,
    load_actor_prior,
)


def _actor_state(input_dim: int) -> dict[str, torch.Tensor]:
    return {
        "distribution.std_param": torch.tensor([1.6]),
        "mlp.0.weight": torch.arange(256 * input_dim, dtype=torch.float32).reshape(256, input_dim),
        "mlp.0.bias": torch.arange(256, dtype=torch.float32),
        "mlp.2.weight": torch.arange(128 * 256, dtype=torch.float32).reshape(128, 256),
        "mlp.2.bias": torch.arange(128, dtype=torch.float32),
        "mlp.4.weight": torch.arange(128 * 128, dtype=torch.float32).reshape(128, 128),
        "mlp.4.bias": torch.arange(128, dtype=torch.float32),
        "mlp.6.weight": torch.arange(6 * 128, dtype=torch.float32).reshape(6, 128),
        "mlp.6.bias": torch.arange(6, dtype=torch.float32),
    }


class _Distribution(torch.nn.Module):
    std_param: torch.nn.Parameter

    def __init__(self) -> None:
        super().__init__()
        self.std_param = torch.nn.Parameter(torch.tensor([0.7]))


class _WalkingActor(torch.nn.Module):
    mlp: torch.nn.Sequential
    distribution: _Distribution

    def __init__(self, input_dim: int = WALKING_OBSERVATION_DIM) -> None:
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 6),
        )
        self.distribution = _Distribution()


def test_build_walking_prior_maps_only_semantically_shared_observations(tmp_path):
    source = tmp_path / "balance.pt"
    output = tmp_path / "walking_prior.pt"
    source_actor = _actor_state(BALANCE_OBSERVATION_DIM)
    torch.save(
        {
            "actor_state_dict": source_actor,
            "critic_state_dict": {"not": torch.tensor(1.0)},
            "optimizer_state_dict": {"state": {}},
            "iter": 200,
        },
        source,
    )

    metadata = build_walking_actor_prior(source, output)
    prior = torch.load(output, map_location="cpu", weights_only=False)
    mapped = prior["actor_mlp_state_dict"]

    assert metadata["transfer_scope"] == "balance_to_walking_actor_mlp_only"
    assert metadata["source_iteration"] == 200
    assert "critic_state_dict" not in prior
    assert "optimizer_state_dict" not in prior
    assert "distribution.std_param" not in mapped
    torch.testing.assert_close(mapped["mlp.0.weight"][:, :39], source_actor["mlp.0.weight"][:, :39])
    torch.testing.assert_close(mapped["mlp.0.weight"][:, 39:], torch.zeros(256, 5))
    torch.testing.assert_close(mapped["mlp.2.weight"], source_actor["mlp.2.weight"])
    torch.testing.assert_close(mapped["mlp.6.weight"], source_actor["mlp.6.weight"])


def test_build_frequency_balance_prior_inserts_frequency_before_support_features(tmp_path):
    source = tmp_path / "balance.pt"
    output = tmp_path / "frequency_balance.pt"
    source_actor = _actor_state(BALANCE_OBSERVATION_DIM)
    torch.save({"actor_state_dict": source_actor, "iter": 200}, source)

    metadata = build_frequency_balance_actor_prior(source, output)
    mapped = torch.load(output, map_location="cpu", weights_only=False)["actor_mlp_state_dict"]["mlp.0.weight"]

    assert metadata["target_observation_dim"] == FREQUENCY_BALANCE_OBSERVATION_DIM
    torch.testing.assert_close(mapped[:, :39], source_actor["mlp.0.weight"][:, :39])
    torch.testing.assert_close(mapped[:, 39], torch.zeros(256))
    torch.testing.assert_close(mapped[:, 40:42], source_actor["mlp.0.weight"][:, 39:41])


def test_build_walking_prior_maps_frequency_balance_command_semantics(tmp_path):
    source = tmp_path / "frequency_balance.pt"
    output = tmp_path / "walking.pt"
    source_actor = _actor_state(FREQUENCY_BALANCE_OBSERVATION_DIM)
    torch.save({"actor_state_dict": source_actor, "iter": 300}, source)

    metadata = build_walking_actor_prior(source, output)
    mapped = torch.load(output, map_location="cpu", weights_only=False)["actor_mlp_state_dict"]["mlp.0.weight"]

    assert metadata["transfer_scope"] == "frequency_balance_to_frequency_walking_actor_mlp_only"
    assert metadata["target_observation_dim"] == FREQUENCY_COMMAND_OBSERVATION_DIM
    torch.testing.assert_close(mapped[:, :39], source_actor["mlp.0.weight"][:, :39])
    torch.testing.assert_close(mapped[:, 39:44], torch.zeros(256, 5))
    torch.testing.assert_close(mapped[:, 44:46], source_actor["mlp.0.weight"][:, 39:41])


def test_build_walking_prior_preserves_all_walking_observation_columns(tmp_path):
    source = tmp_path / "walking.pt"
    output = tmp_path / "walking_prior.pt"
    source_actor = _actor_state(WALKING_OBSERVATION_DIM)
    torch.save({"actor_state_dict": source_actor, "iter": 650}, source)

    metadata = build_walking_actor_prior(source, output)
    prior = torch.load(output, map_location="cpu", weights_only=False)

    assert metadata["transfer_scope"] == "walking_actor_mlp_only"
    assert metadata["source_iteration"] == 650
    assert metadata["excluded_balance_observations"] == {}
    torch.testing.assert_close(prior["actor_mlp_state_dict"]["mlp.0.weight"], source_actor["mlp.0.weight"])


def test_build_periodic_base_prior_drops_phase_columns_with_equal_input_width(tmp_path):
    source = tmp_path / "periodic_base.pt"
    output = tmp_path / "walking_prior.pt"
    source_actor = _actor_state(PERIODIC_BASE_OBSERVATION_DIM)
    torch.save({"actor_state_dict": source_actor, "iter": 400}, source)

    metadata = build_periodic_base_walking_actor_prior(source, output)
    mapped = torch.load(output, map_location="cpu", weights_only=False)["actor_mlp_state_dict"]["mlp.0.weight"]

    assert metadata["transfer_scope"] == "periodic_base_to_walking_actor_mlp_only"
    assert metadata["source_iteration"] == 400
    torch.testing.assert_close(mapped[:, :39], source_actor["mlp.0.weight"][:, :39])
    torch.testing.assert_close(mapped[:, 39:], torch.zeros(256, 5))


def test_build_periodic_base_continuation_prior_preserves_all_actor_inputs(tmp_path):
    source = tmp_path / "periodic_base.pt"
    output = tmp_path / "periodic_walking_prior.pt"
    source_actor = _actor_state(PERIODIC_BASE_OBSERVATION_DIM)
    torch.save({"actor_state_dict": source_actor, "iter": 400}, source)

    metadata = build_periodic_base_continuation_actor_prior(source, output)
    prior = torch.load(output, map_location="cpu", weights_only=False)

    assert metadata["transfer_scope"] == "periodic_base_actor_mlp_full_continuation"
    assert metadata["target_observation_dim"] == PERIODIC_BASE_OBSERVATION_DIM
    torch.testing.assert_close(prior["actor_mlp_state_dict"]["mlp.0.weight"], source_actor["mlp.0.weight"])


def test_build_frequency_command_prior_preserves_command_observation_columns(tmp_path):
    source = tmp_path / "frequency_command.pt"
    output = tmp_path / "frequency_command_prior.pt"
    source_actor = _actor_state(FREQUENCY_COMMAND_OBSERVATION_DIM)
    torch.save({"actor_state_dict": source_actor, "iter": 1000}, source)

    metadata = build_walking_actor_prior(source, output)
    prior = torch.load(output, map_location="cpu", weights_only=False)

    assert metadata["transfer_scope"] == "frequency_command_actor_mlp_only"
    assert metadata["target_observation_dim"] == FREQUENCY_COMMAND_OBSERVATION_DIM
    assert metadata["walking_only_observations"]["target_support_side"] == [45, 46]
    torch.testing.assert_close(prior["actor_mlp_state_dict"]["mlp.0.weight"], source_actor["mlp.0.weight"])


def test_load_walking_prior_preserves_fresh_exploration_state(tmp_path):
    source = tmp_path / "balance.pt"
    prior_path = tmp_path / "walking_prior.pt"
    torch.save({"actor_state_dict": _actor_state(BALANCE_OBSERVATION_DIM), "iter": 200}, source)
    build_walking_actor_prior(source, prior_path)

    actor = _WalkingActor()
    initial_std = actor.distribution.std_param.detach().clone()
    runner = SimpleNamespace(alg=SimpleNamespace(get_policy=lambda: actor))
    metadata = load_actor_prior(runner, prior_path)

    assert metadata["reward_alignment"].startswith("critic, optimizer")
    torch.testing.assert_close(actor.distribution.std_param, initial_std)
    first_layer = cast(torch.nn.Linear, cast(torch.nn.Sequential, actor.mlp)[0])
    torch.testing.assert_close(first_layer.weight[:, 39:], torch.zeros(256, 5))


def test_load_walking_prior_zero_initializes_frequency_command_column(tmp_path):
    source = tmp_path / "walking.pt"
    prior_path = tmp_path / "walking_prior.pt"
    source_actor = _actor_state(WALKING_OBSERVATION_DIM)
    torch.save({"actor_state_dict": source_actor, "iter": 500}, source)
    build_walking_actor_prior(source, prior_path)

    actor = _WalkingActor(FREQUENCY_COMMAND_OBSERVATION_DIM)
    runner = SimpleNamespace(alg=SimpleNamespace(get_policy=lambda: actor))
    load_actor_prior(runner, prior_path)

    first_layer = cast(torch.nn.Linear, cast(torch.nn.Sequential, actor.mlp)[0])
    torch.testing.assert_close(first_layer.weight[:, :WALKING_OBSERVATION_DIM], source_actor["mlp.0.weight"])
    torch.testing.assert_close(
        first_layer.weight[:, WALKING_OBSERVATION_DIM:FREQUENCY_COMMAND_OBSERVATION_DIM],
        torch.zeros(256, FREQUENCY_COMMAND_OBSERVATION_DIM - WALKING_OBSERVATION_DIM),
    )

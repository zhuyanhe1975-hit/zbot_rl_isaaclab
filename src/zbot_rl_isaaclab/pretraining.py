# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Actor-only transfer from the balance policy into the 6-DoF walking task."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

BALANCE_OBSERVATION_DIM = 41
FREQUENCY_BALANCE_OBSERVATION_DIM = 42
WALKING_OBSERVATION_DIM = 44
PERIODIC_BASE_OBSERVATION_DIM = 44
FREQUENCY_COMMAND_OBSERVATION_DIM = 46
PRIOR_FORMAT_VERSION = 1

# The balance-only command features at [39:41] deliberately have no walking destination.
SHARED_OBSERVATION_SLICES: dict[str, tuple[slice, slice]] = {
    "base_lin_vel": (slice(0, 3), slice(0, 3)),
    "base_ang_vel": (slice(3, 6), slice(3, 6)),
    "projected_gravity": (slice(6, 9), slice(6, 9)),
    "joint_pos": (slice(9, 15), slice(9, 15)),
    "joint_vel": (slice(15, 21), slice(15, 21)),
    "actions": (slice(21, 27), slice(21, 27)),
    "foot_contact_forces": (slice(27, 33), slice(27, 33)),
    "com_relative_to_feet": (slice(33, 39), slice(33, 39)),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_frequency_balance_actor_prior(source_checkpoint: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Map a fixed-period balance actor into the frequency-conditioned balance task."""
    source_path = Path(source_checkpoint).resolve()
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    source_actor = checkpoint["actor_state_dict"]
    source_input = source_actor["mlp.0.weight"]
    if source_input.ndim != 2 or source_input.shape[1] != BALANCE_OBSERVATION_DIM:
        raise ValueError(f"Expected balance actor input width {BALANCE_OBSERVATION_DIM}.")
    target_input = source_input.new_zeros((source_input.shape[0], FREQUENCY_BALANCE_OBSERVATION_DIM))
    target_input[:, :39] = source_input[:, :39]
    target_input[:, 40:42] = source_input[:, 39:41]
    actor_mlp_state = {
        key: value.detach().cpu().clone() for key, value in source_actor.items() if key.startswith("mlp.")
    }
    actor_mlp_state["mlp.0.weight"] = target_input
    metadata = {
        "format": "zbot_walking_actor_prior",
        "version": PRIOR_FORMAT_VERSION,
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": _sha256(source_path),
        "source_iteration": int(checkpoint["iter"]),
        "source_observation_dim": BALANCE_OBSERVATION_DIM,
        "target_observation_dim": FREQUENCY_BALANCE_OBSERVATION_DIM,
        "shared_observation_slices": {"balance_state": {"source": [0, 39], "target": [0, 39]}},
        "transfer_scope": "balance_to_frequency_balance_actor_mlp_only",
        "reward_alignment": "critic, optimizer, iteration, and exploration std start fresh",
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": metadata, "actor_mlp_state_dict": actor_mlp_state}, output)
    return metadata


def build_periodic_base_walking_actor_prior(source_checkpoint: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Map the 44-D periodic Base actor into the semantically different 44-D walking policy."""
    source_path = Path(source_checkpoint).resolve()
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    source_actor = checkpoint["actor_state_dict"]
    source_input = source_actor["mlp.0.weight"]
    if source_input.ndim != 2 or source_input.shape[1] != PERIODIC_BASE_OBSERVATION_DIM:
        raise ValueError(f"Expected periodic Base actor input width {PERIODIC_BASE_OBSERVATION_DIM}.")
    target_input = source_input.new_zeros((source_input.shape[0], WALKING_OBSERVATION_DIM))
    target_input[:, :39] = source_input[:, :39]
    actor_mlp_state = {
        key: value.detach().cpu().clone() for key, value in source_actor.items() if key.startswith("mlp.")
    }
    actor_mlp_state["mlp.0.weight"] = target_input
    metadata = {
        "format": "zbot_walking_actor_prior",
        "version": PRIOR_FORMAT_VERSION,
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": _sha256(source_path),
        "source_iteration": int(checkpoint["iter"]),
        "source_observation_dim": PERIODIC_BASE_OBSERVATION_DIM,
        "target_observation_dim": WALKING_OBSERVATION_DIM,
        "shared_observation_slices": {
            name: {"source": [source.start, source.stop], "target": [target.start, target.stop]}
            for name, (source, target) in SHARED_OBSERVATION_SLICES.items()
        },
        "excluded_balance_observations": {
            "frequency": [39, 40],
            "target_support_side": [40, 41],
            "phase_sin_cos": [41, 43],
            "support_distance_error": [43, 44],
        },
        "walking_only_observations": {
            "heading_error": [39, 40],
            "base_height": [40, 41],
            "foot_contacts": [41, 43],
            "joint_velocity_limit": [43, 44],
        },
        "transfer_scope": "periodic_base_to_walking_actor_mlp_only",
        "reward_alignment": "critic, optimizer, iteration, and exploration std are initialized from walking config",
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": metadata, "actor_mlp_state_dict": actor_mlp_state}, output)
    return metadata


def build_periodic_base_continuation_actor_prior(
    source_checkpoint: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Preserve every Base actor input and MLP parameter for periodic walking continuation."""
    source_path = Path(source_checkpoint).resolve()
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    source_actor = checkpoint["actor_state_dict"]
    source_input = source_actor["mlp.0.weight"]
    if source_input.ndim != 2 or source_input.shape[1] != PERIODIC_BASE_OBSERVATION_DIM:
        raise ValueError(f"Expected periodic Base actor input width {PERIODIC_BASE_OBSERVATION_DIM}.")
    actor_mlp_state = {
        key: value.detach().cpu().clone() for key, value in source_actor.items() if key.startswith("mlp.")
    }
    metadata = {
        "format": "zbot_walking_actor_prior",
        "version": PRIOR_FORMAT_VERSION,
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": _sha256(source_path),
        "source_iteration": int(checkpoint["iter"]),
        "source_observation_dim": PERIODIC_BASE_OBSERVATION_DIM,
        "target_observation_dim": PERIODIC_BASE_OBSERVATION_DIM,
        "shared_observation_slices": {"periodic_base_policy": {"source": [0, 44], "target": [0, 44]}},
        "excluded_balance_observations": {},
        "walking_only_observations": {},
        "transfer_scope": "periodic_base_actor_mlp_full_continuation",
        "reward_alignment": (
            "critic, optimizer, iteration, and exploration std are initialized from periodic walking config"
        ),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": metadata, "actor_mlp_state_dict": actor_mlp_state}, output)
    return metadata


def build_walking_actor_prior(source_checkpoint: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Create a walking-compatible actor prior from a balance-task RSL-RL checkpoint."""
    source_path = Path(source_checkpoint).resolve()
    output = Path(output_path)
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    source_actor = checkpoint["actor_state_dict"]
    source_input = source_actor["mlp.0.weight"]
    if source_input.ndim != 2 or source_input.shape[1] not in (
        BALANCE_OBSERVATION_DIM,
        FREQUENCY_BALANCE_OBSERVATION_DIM,
        WALKING_OBSERVATION_DIM,
        FREQUENCY_COMMAND_OBSERVATION_DIM,
    ):
        raise ValueError(
            f"Expected actor input width {BALANCE_OBSERVATION_DIM}, {FREQUENCY_BALANCE_OBSERVATION_DIM}, "
            f"{WALKING_OBSERVATION_DIM}, "
            f"or {FREQUENCY_COMMAND_OBSERVATION_DIM}, "
            f"got {tuple(source_input.shape)}."
        )

    source_observation_dim = int(source_input.shape[1])
    if source_observation_dim in (WALKING_OBSERVATION_DIM, FREQUENCY_COMMAND_OBSERVATION_DIM):
        target_input = source_input.detach().cpu().clone()
        transfer_scope = (
            "frequency_command_actor_mlp_only"
            if source_observation_dim == FREQUENCY_COMMAND_OBSERVATION_DIM
            else "walking_actor_mlp_only"
        )
    elif source_observation_dim == FREQUENCY_BALANCE_OBSERVATION_DIM:
        target_input = source_input.new_zeros((source_input.shape[0], FREQUENCY_COMMAND_OBSERVATION_DIM))
        target_input[:, :39] = source_input[:, :39]
        target_input[:, 44:46] = source_input[:, 39:41]
        transfer_scope = "frequency_balance_to_frequency_walking_actor_mlp_only"
    else:
        target_input = source_input.new_zeros((source_input.shape[0], WALKING_OBSERVATION_DIM))
        for source_slice, target_slice in SHARED_OBSERVATION_SLICES.values():
            target_input[:, target_slice] = source_input[:, source_slice]
        transfer_scope = "balance_to_walking_actor_mlp_only"

    actor_mlp_state = {
        key: value.detach().cpu().clone() for key, value in source_actor.items() if key.startswith("mlp.")
    }
    actor_mlp_state["mlp.0.weight"] = target_input
    metadata = {
        "format": "zbot_walking_actor_prior",
        "version": PRIOR_FORMAT_VERSION,
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": _sha256(source_path),
        "source_iteration": int(checkpoint["iter"]),
        "source_observation_dim": source_observation_dim,
        "target_observation_dim": int(target_input.shape[1]),
        "shared_observation_slices": {
            name: {
                "source": [source.start, source.stop],
                "target": [target.start, target.stop],
            }
            for name, (source, target) in SHARED_OBSERVATION_SLICES.items()
        },
        "excluded_balance_observations": (
            {
                "ideal_support_foot": [39, 40],
                "support_distance_error": [40, 41],
            }
            if source_observation_dim == BALANCE_OBSERVATION_DIM
            else {"support_distance_error": [41, 42]}
            if source_observation_dim == FREQUENCY_BALANCE_OBSERVATION_DIM
            else {}
        ),
        "walking_only_observations": {
            "heading_error": [39, 40],
            "base_height": [40, 41],
            "foot_contacts": [41, 43],
            "joint_velocity_limit": [43, 44],
        }
        | (
            {
                "step_frequency": [44, 45],
                "target_support_side": [45, 46],
            }
            if source_observation_dim in (FREQUENCY_BALANCE_OBSERVATION_DIM, FREQUENCY_COMMAND_OBSERVATION_DIM)
            else {}
        ),
        "transfer_scope": transfer_scope,
        "reward_alignment": "critic, optimizer, iteration, and exploration std are initialized from walking config",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": metadata, "actor_mlp_state_dict": actor_mlp_state}, output)
    return metadata


def load_actor_prior(runner: Any, prior_path: str | Path) -> dict[str, Any]:
    """Load only compatible actor MLP weights, leaving walking reward state fresh."""
    prior = torch.load(prior_path, map_location="cpu", weights_only=False)
    metadata = prior.get("metadata", {})
    if metadata.get("format") != "zbot_walking_actor_prior" or metadata.get("version") != PRIOR_FORMAT_VERSION:
        raise ValueError(f"Unsupported walking actor prior: {prior_path}")

    actor = runner.alg.get_policy()
    target_state = actor.state_dict()
    transferred: Mapping[str, torch.Tensor] = prior["actor_mlp_state_dict"]
    for key, source_value in transferred.items():
        if key not in target_state:
            raise KeyError(f"Walking actor has no parameter {key!r} required by the prior.")
        if (
            key == "mlp.0.weight"
            and source_value.shape[1] == WALKING_OBSERVATION_DIM
            and target_state[key].shape[0] == source_value.shape[0]
            and target_state[key].shape[1] == FREQUENCY_COMMAND_OBSERVATION_DIM
        ):
            adapted_value = torch.zeros_like(target_state[key])
            adapted_value[:, :WALKING_OBSERVATION_DIM] = source_value.to(
                device=adapted_value.device,
                dtype=adapted_value.dtype,
            )
            target_state[key] = adapted_value
            continue
        if target_state[key].shape != source_value.shape:
            raise ValueError(
                f"Walking actor parameter {key!r} has shape {tuple(target_state[key].shape)}, "
                f"but the prior has {tuple(source_value.shape)}."
            )
        target_state[key] = source_value.to(device=target_state[key].device, dtype=target_state[key].dtype)
    actor.load_state_dict(target_state, strict=True)
    return metadata


def install_actor_prior(prior_path: str | Path) -> None:
    """Install a one-process hook that initializes each constructed OnPolicyRunner actor."""
    from rsl_rl.runners import OnPolicyRunner

    original_init = OnPolicyRunner.__init__
    resolved_path = str(Path(prior_path).resolve())

    def prior_initialized_runner(self, *args, **kwargs) -> None:
        original_init(self, *args, **kwargs)
        metadata = load_actor_prior(self, resolved_path)
        source_dim = metadata["source_observation_dim"]
        transfer_detail = (
            f"all {source_dim} walking observation columns"
            if source_dim == WALKING_OBSERVATION_DIM
            else f"{len(metadata['shared_observation_slices'])} aligned observation terms"
        )
        print(
            f"[INFO] Loaded walking actor prior from iteration {metadata['source_iteration']} with {transfer_detail}."
        )
        print("[INFO] Walking critic, optimizer, iteration, and exploration std start fresh.")

    OnPolicyRunner.__init__ = prior_initialized_runner

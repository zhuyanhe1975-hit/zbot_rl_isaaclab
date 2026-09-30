#!/usr/bin/env python3
"""Build a walking-compatible actor prior from a balance checkpoint."""

import argparse

from zbot_rl_isaaclab.pretraining import (
    build_frequency_balance_actor_prior,
    build_periodic_base_continuation_actor_prior,
    build_periodic_base_walking_actor_prior,
    build_walking_actor_prior,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_checkpoint")
    parser.add_argument("output_path")
    parser.add_argument("--frequency-balance", action="store_true")
    parser.add_argument("--periodic-base", action="store_true")
    parser.add_argument("--periodic-base-continuation", action="store_true")
    args = parser.parse_args()
    selected_modes = sum((args.frequency_balance, args.periodic_base, args.periodic_base_continuation))
    if selected_modes > 1:
        parser.error("Prior source-layout flags are mutually exclusive.")
    if args.frequency_balance:
        builder = build_frequency_balance_actor_prior
    elif args.periodic_base:
        builder = build_periodic_base_walking_actor_prior
    elif args.periodic_base_continuation:
        builder = build_periodic_base_continuation_actor_prior
    else:
        builder = build_walking_actor_prior
    metadata = builder(args.source_checkpoint, args.output_path)
    print(f"Saved walking actor prior to {args.output_path}")
    print(f"Transferred observation terms: {', '.join(metadata['shared_observation_slices'])}")


if __name__ == "__main__":
    main()

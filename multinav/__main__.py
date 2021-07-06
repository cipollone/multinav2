#!/usr/bin/env python3

"""Main script."""

import argparse
import logging
import random

import numpy as np
import yaml


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="RL on navigation environments with different abstractions"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print debug information",
    )

    # What to do
    parser.add_argument(
        "do",
        choices=["train", "test"],
        help="What to do",
    )

    # Options
    parser.add_argument(
        "-p",
        "--params",
        type=str,
        required=True,
        help="Json file of parameters."
    )

    # Parse
    args = parser.parse_args()

    # logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Load params
    with open(args.params) as f:
        params = yaml.safe_load(f)

    # Set seed
    set_seed(params["seed"])

    # Go
    if args.do == "train":
        from multinav import training

        training.train(params)

    elif args.do == "test":
        from multinav import testing

        testing.test(params)


def set_seed(seed: int):
    """Set seed."""
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    main()

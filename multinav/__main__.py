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
    ops = parser.add_subparsers(dest="do", help="what to do")

    # Training
    train_parser = ops.add_parser("train")
    train_parser.add_argument(
        "-p",
        "--params",
        type=str,
        required=True,
        help="yaml file of parameters."
    )

    # Testing
    test_parser = ops.add_parser("test")
    test_parser.add_argument(
        "-p",
        "--params",
        type=str,
        required=True,
        help="yaml file of parameters of a previous training.",
    )
    test_parser.add_argument(
        "-l",
        "--load",
        type=str,
        required=True,
        help="path of an agent checkpoint",
    )
    test_parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Interact with the agent",
    )
    test_parser.add_argument(
        "-r",
        "--render",
        action="store_true",
        help="Render the image",
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

        training.TrainerSetup(params).train()

    elif args.do == "test":
        from multinav import testing

        testing.test(params, args.load, args.interactive)


def set_seed(seed: int):
    """Set seed."""
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    main()

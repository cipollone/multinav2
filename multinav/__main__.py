#!/usr/bin/env python3

"""Main script."""

import argparse
import logging

import time
import random
import numpy as np
import tensorflow as tf


def main():
    """Is the main function."""
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
        "-e",
        "--env",
        required=True,
        choices=["sapientino-abs", "sapientino-grid", "sapientino-cont", "ros"],
        help="Select the environment to use",
    )
    parser.add_argument(
        "-p",
        "--params",
        help="Json file of parameters. "
        "Launch a training without this  to generate a skeleton",
    )
    parser.add_argument(
        "-r", "--resume", help="Resume from checkpoint file. Overrides json params."
    )
    parser.add_argument(
        "-s",
        "--shaping",
        help="Apply reward shaping from checkpoint of the more abstract environment.",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Test with an interactive episode (used for debugging).",
    )
    parser.add_argument(
        "-d",
        "--deterministic",
        action="store_true",
        help="Test with a deterministic policy",
    )
    parser.add_argument(
        "--passive", action="store_true", help="Test the passive agent",
    )
    parser.add_argument(
        "--id", type=int, help="run id: use this for multiple runs"
    )
    parser.add_argument(
        "--seed", type=int, help="Manually set a random seed."
    )

    # Parse
    args = parser.parse_args()

    # logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    seed = init_seed(args.seed)

    # Other params
    cmd_params = dict()
    if args.resume:
        cmd_params["resume_file"] = args.resume
    if args.shaping:
        cmd_params["shaping"] = args.shaping
    cmd_params["interactive"] = args.interactive
    cmd_params["deterministic"] = args.deterministic
    cmd_params["test_passive"] = args.passive
    cmd_params["run_id"] = args.id
    cmd_params["seed"] = seed

    # Go
    if args.do == "train":
        from multinav import training

        training.train(
            env_name=args.env,
            json_params=args.params,
            cmd_params=cmd_params,
        )

    elif args.do == "test":
        from multinav import testing

        testing.test(
            env_name=args.env,
            json_params=args.params,
            cmd_params=cmd_params,
        )


def init_seed(seed=None):
    """Sample a seed or use the one given."""

    if seed is None:
        seed = int(time.time())

    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)  # NOTE: tf random ops have their own seed.

    return seed


if __name__ == "__main__":
    main()

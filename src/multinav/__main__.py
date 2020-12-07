#!/usr/bin/env python3

"""Main script."""

import argparse

from multinav import training
from multinav.envs.ros_controls import RosControlsEnv


def main():
    """Is the main function."""
    parser = argparse.ArgumentParser(
        description="RL on navigation with different abstractions"
    )
    op_group = parser.add_subparsers(dest="op", help="What to do")

    # Train an agent
    train_parser = op_group.add_parser("train")
    train_parser.add_argument(
        "-e",
        "--env",
        choices=["ros"],
        help="Select the environment to train the agent on.",
    )

    # Test ros
    _ = op_group.add_parser("test-ros")

    # Parse
    args = parser.parse_args()

    # Go
    if args.op is None:
        print(parser.print_help())

    if args.op == "test-ros":
        RosControlsEnv.interactive_test()

    elif args.op == "train":
        if args.env == "ros":
            training.train_on_ros()


if __name__ == "__main__":
    main()

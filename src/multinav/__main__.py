#!/usr/bin/env python3

"""Main script."""

import argparse

from multinav import training, testing
from multinav.envs.ros_controls import interactive_test


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
    train_parser.add_argument(
        "-p",
        "--params",
        help="Json file of parameters",
    )

    # Test an agent
    test_parser = op_group.add_parser("test")
    test_parser.add_argument(
        "-e",
        "--env",
        choices=["ros"],
        required=True,
        help="Select the environment to train the agent on.",
    )
    test_parser.add_argument(
        "-p",
        "--params",
        required=True,
        help="Json file of parameters",
    )

    # Test ros
    _ = op_group.add_parser("test-ros")

    # Parse
    args = parser.parse_args()

    # Go
    if args.op is None:
        print(parser.print_help())

    if args.op == "test-ros":
        interactive_test()

    elif args.op == "train":
        if args.env == "ros":
            training.train_on_ros(json_args=args.params)

    elif args.op == "test":
        if args.env == "ros":
            testing.test_on_ros(json_args=args.params)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""Main script."""

import argparse

from multinav.envs.ros_controls import RosControlsEnv


def main():
    """Main function."""

    parser = argparse.ArgumentParser(
        description="RL on navigation with different abstractions")

    # What to do
    parser.add_argument("op", choices=["test-ros"], help="What to do")

    args = parser.parse_args()

    # Do
    if args.op == "test-ros":
        RosControlsEnv.interactive_test()


if __name__ == "__main__":
    main()

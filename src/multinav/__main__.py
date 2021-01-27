#!/usr/bin/env python3

"""Main script."""

import argparse
import logging


def main():
    """Is the main function."""
    parser = argparse.ArgumentParser(
        description="RL on navigation environments with different abstractions"
    )
    parser.add_argument(
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
        "--deterministic",
        action="store_true",
        help="Test with a deterministic policy",
    )

    # Parse
    args = parser.parse_args()

    # logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Other params
    cmd_params = dict()
    if args.resume:
        cmd_params["resume_file"] = args.resume
    if args.shaping:
        cmd_params["shaping"] = args.shaping
    cmd_params["interactive"] = args.interactive
    cmd_params["deterministic"] = args.deterministic

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


if __name__ == "__main__":
    main()

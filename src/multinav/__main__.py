#!/usr/bin/env python3

"""Main script."""

import argparse

# TODO: normalize the environment features. Look in 736e60d for example.


def main():
    """Is the main function."""
    parser = argparse.ArgumentParser(
        description="RL on navigation environments with different abstractions"
    )

    # Args
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
        "do",
        choices=["train", "test"],
        help="What to do",
    )

    # Parse
    args = parser.parse_args()

    # Other params
    cmd_params = dict()
    if args.resume:
        cmd_params["resume_file"] = args.resume

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

"""
Training CLI entrypoint.
Currently only a bootstrap placeholder.
"""

import argparse
from aiflab.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ailab-train",
        description="Run Active Inference experiments",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to experiment config",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    configure_logging()

    print("AILab bootstrap CLI")
    print(f"config={args.config}")
    print(f"seed={args.seed}")
    print(f"episodes={args.episodes}")


if __name__ == "__main__":
    main()
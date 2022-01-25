#!/usr/env/bin python

r"""
Combine multiple metric files
"""

import argparse
import pathlib

import yaml


def parse_args() -> argparse.Namespace:
    r"""
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Combine multiple metric files"
    )
    parser.add_argument(
        "-i", "--input", dest="input", type=pathlib.Path, nargs="+",
        help="Path of input metric files (.yaml)"
    )
    parser.add_argument(
        "-o", "--output", dest="output", type=pathlib.Path, required=True,
        help="Path of output metric file (.yaml)"
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    r"""
    Main function
    """
    print("[1/2] Reading input...")
    metrics = {}
    for input in args.input:
        with input.open("r") as f:
            metrics.update(yaml.load(f, Loader=yaml.Loader))
    metrics.pop("args", None)

    print("[2/2] Writing output...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        yaml.dump(metrics, f)


if __name__ == "__main__":
    main(parse_args())

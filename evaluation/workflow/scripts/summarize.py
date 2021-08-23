#!/usr/bin/env python

r"""
Summarize metric files
"""

import argparse
from typing import Union

import pandas as pd
import parse
import yaml
from snakemake.script import Snakemake


def parse_args() -> argparse.Namespace:
    r"""
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Summarize metric files into a single data frame"
    )
    parser.add_argument(
        "-i", "--input", dest="input", type=str, nargs="+",
        help="Path of input metric files (.yaml)"
    )
    parser.add_argument(
        "-p", "--pattern", dest="pattern", type=str, required=True,
        help="File path pattern used for extracting relevant meta information"
    )
    parser.add_argument(
        "-o", "--output", dest="output", type=str, required=True,
        help="Path of output summary file (.csv)"
    )
    args = parser.parse_args()
    args.params = argparse.Namespace(pattern=args.pattern)
    args.output = [args.output]
    del args.pattern
    return args


def main(snakemake: Union[argparse.Namespace, Snakemake]) -> None:  # pylint: disable=redefined-outer-name
    r"""
    Main function
    """
    df = []
    for item in set(snakemake.input):
        entry = parse.parse(snakemake.params.pattern, item)
        if entry:
            conf = entry.named
        else:
            continue
        with open(item, "r") as f:
            performance = yaml.load(f, Loader=yaml.Loader)
            performance.pop("cmd", None)  # Discard, can exist in run_info.yaml
            performance.pop("args", None)  # Discard, can exist in run_info.yaml
        df.append({**conf, **performance})
    sort_order = list(conf.keys())
    df = pd.DataFrame.from_records(df).sort_values(sort_order)
    df.to_csv(snakemake.output[0], index=False)


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = parse_args()
    main(snakemake)

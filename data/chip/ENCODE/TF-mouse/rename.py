#!/usr/bin/env python

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", type=Path, required=True)
    parser.add_argument("-o", "--output", dest="output", type=Path, required=True)
    return parser.parse_args()


def main(args):
    bed = pd.read_table(
        args.input, header=None,
        names=["chrom", "chromStart", "chromEnd", "name"]
    )
    ortholog = pd.read_csv(
        "../../../genome/human-mouse-orthologs.csv.gz",
        usecols=["Gene name", "Mouse gene name"]
    ).drop_duplicates()

    human_count = ortholog["Gene name"].value_counts()
    human_uniq = set(human_count.index[human_count == 1])
    ortholog = ortholog.loc[[item in human_uniq for item in ortholog["Gene name"]]]
    mapping = {row["Gene name"]: row["Mouse gene name"] for _, row in ortholog.iterrows()}
    bed["name"] = bed["name"].map(mapping)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    bed.to_csv(args.output, sep="\t", header=False, index=False)


if __name__ == "__main__":
    main(parse_args())

#!/usr/bin/env python

r"""
Clean motif hits to a target species
"""

from argparse import ArgumentParser

import pandas as pd
from scglue.genomics import read_gtf


def parse_args():
    r"""
    Parse command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", type=str, required=True)
    parser.add_argument("-a", "--annotation", dest="annotation", type=str, required=True)
    parser.add_argument("-t", "--orthologs", dest="orthologs", type=str, nargs="*", default=[])
    parser.add_argument("-o", "--output", dest="output", type=str, required=True)
    return parser.parse_args()


def main(args):
    r"""
    Main function
    """
    fimo = pd.read_csv(args.input)
    annotation = read_gtf(args.annotation).split_attribute()
    annotation = {i: i for i in set(annotation["gene_name"])}
    ortholog = pd.concat([
        pd.read_csv(ortholog, header=None).dropna()
        for ortholog in args.orthologs
    ])
    ortholog_count = ortholog[0].value_counts()
    ortholog_use = set(ortholog_count.index[ortholog_count == 1])
    ortholog = {
        i: j for i, j in zip(ortholog[0], ortholog[1])
        if i in ortholog_use and j != i
    }
    combined_map = {**annotation, **ortholog}
    fimo["motif_alt_id"] = fimo["motif_alt_id"].map(combined_map).map(annotation)
    fimo = fimo.dropna(subset=["motif_alt_id"])
    fimo.to_csv(args.output, index=False)


if __name__ == "__main__":
    main(parse_args())

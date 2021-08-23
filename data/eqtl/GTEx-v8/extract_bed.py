#!/usr/bin/env python

import argparse
import pathlib

import pandas as pd

import scglue


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", type=pathlib.Path, required=True)
    parser.add_argument("-o", "--output", dest="output", type=pathlib.Path, required=True)
    return parser.parse_args()


def main(args):
    df = pd.read_table(args.input)
    df["variant_info"] = df["variant_id"].str.split("_")
    df["chrom"] = df["variant_info"].map(lambda x: x[0])
    df["chromStart"] = df["variant_info"].map(lambda x: int(x[1]) - 1)
    df["chromEnd"] = df["variant_info"].map(lambda x: len(x[2]))
    df["chromEnd"] += df["chromStart"]
    df["name"] = df["gene_id"]
    df["score"] = df["pval_beta"]
    bed = scglue.genomics.Bed(df)
    bed.write_bed(args.output, ncols=5)


if __name__ == "__main__":
    main(parse_args())


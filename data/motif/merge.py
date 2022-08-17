#!/usr/bin/env python

r"""
Merge fimo hits per TF
"""

from argparse import ArgumentParser
from multiprocessing import Pool

import pandas as pd
import pybedtools
from pybedtools import BedTool
from pybedtools.cbedtools import Interval
from tqdm import tqdm


def to_bedtool(df):
    r"""
    Convert pandas data frame to bedtool object
    """
    return BedTool(Interval(
        row["chrom"], row["chromStart"], row["chromEnd"], name=row["name"]
    ) for _, row in df.iterrows())


def job(args):
    r"""
    Multiprocessing job
    """
    name, fimo = args
    result = to_bedtool(fimo).sort().merge().to_dataframe().assign(name=name)
    return result


def parse_args():
    r"""
    Parse command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", type=str, required=True)
    parser.add_argument("-o", "--output", dest="output", type=str, required=True)
    parser.add_argument("-j", "--n-jobs", dest="n_jobs", type=int, default=1)
    return parser.parse_args()


def main(args):
    r"""
    Main function
    """
    fimo = pd.read_csv(args.input, usecols=["motif_alt_id", "sequence_name", "start", "stop"])
    fimo.columns = ["name", "chrom", "chromStart", "chromEnd"]
    fimo = fimo.loc[:, ["chrom", "chromStart", "chromEnd", "name"]]
    fimo_gb = list(fimo.groupby("name"))
    with Pool(args.n_jobs) as pool:
        result = pd.concat(tqdm(pool.imap(job, fimo_gb), total=len(fimo_gb)))
    result = result.sort_values(["chrom", "start"])
    result.to_csv(args.output, sep="\t", header=False, index=False)
    pybedtools.cleanup(remove_all=True)


if __name__ == "__main__":
    main(parse_args())

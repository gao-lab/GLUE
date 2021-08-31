#!/usr/bin/env python

r"""
Extract genes with promoters BED files from GTF file
"""

import argparse
import pathlib

import scglue


def parse_args() -> argparse.Namespace:
    r"""
    Parse command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-gtf", dest="input_gtf", type=pathlib.Path, required=True,
        help="Path to input GTF file"
    )
    parser.add_argument(
        "--promoter-len", dest="promoter_len", type=int, default=2000,
        help="Promoter length"
    )
    parser.add_argument(
        "--output-bed", dest="output_bed", type=pathlib.Path, required=True,
        help="Path to output BED file"
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
    r"""
    Main function
    """
    gtf = scglue.genomics.read_gtf(args.input_gtf).query("feature == 'gene'").split_attribute()
    bed = gtf.to_bed(name="gene_name").expand(args.promoter_len, 0)
    bed = bed.drop_duplicates(subset=["chrom", "chromStart", "chromEnd"])
    bed.write_bed(args.output_bed, ncols=6)


if __name__ == "__main__":
    main(parse_args())

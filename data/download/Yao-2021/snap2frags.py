import argparse
import pathlib

import h5py
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", type=pathlib.Path, required=True)
    parser.add_argument("-o", "--output", dest="output", type=pathlib.Path, required=True)
    return parser.parse_args()


def main(args):
    with h5py.File(args.input, "r") as f:
        barcode = np.repeat(f["BD"]["name"][:].astype(str), f["FM"]["barcodeLen"][:])
        chrom = f["FM"]["fragChrom"][:].astype(str)
        chromStart = f["FM"]["fragStart"][:]
        chromEnd = chromStart + f["FM"]["fragLen"][:]
    bed = pd.DataFrame({
        "chrom": chrom,
        "chromStart": chromStart,
        "chromEnd": chromEnd,
        "barcode": barcode
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bed.to_csv(args.output, sep="\t", index=False, header=False)


if __name__ == "__main__":
    main(parse_args())

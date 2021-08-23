# Data preparation

## Step 1: Download and preprocessing

The following directories require data download and preprocessing:

```
.
├─ genome
├─ conservation
├─ download
│  ├─ 10x-ATAC-Brain5k
│  ├─ 10x-Multiome-Pbmc10k
│  ├─ Cao-2020
│  ├─ Chen-2019
│  ├─ Domcke-2020
│  ├─ Luo-2017
│  ├─ Ma-2020
│  └─ Saunders-2018
├─ eqtl
│  └─ GTEx-v8
├─ hic
│  └─ Javierre-2016
├─ chip
│  └─ ENCODE
│     └─ TF-human
└─ database
   └─ TRRUST-v2
```

For each of these directories, download the necessary files specified in
README, then run the `preprocess.sh` script.

## Step 2: Format single-cell data

Run the following command to collect single-cell data into a unified
`h5ad` format:

```sh
snakemake -j4 -prk
```

This produces a `dataset` directory containing `*.h5ad` files.

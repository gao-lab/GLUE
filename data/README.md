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

# Direct download

The preprocessed `*.h5ad` data files can also be downloaded directly
from [here](https://scglue.readthedocs.io/en/latest/data.html).

Additionally, for the "FRAGS2RNA" files used in the evaluation pipeline,
please use the following links:

* http://download.gao-lab.org/GLUE/dataset/Chen-2019-FRAGS2RNA.h5ad
* http://download.gao-lab.org/GLUE/dataset/Ma-2020-FRAGS2RNA.h5ad
* http://download.gao-lab.org/GLUE/dataset/10x-Multiome-Pbmc10k-FRAGS2RNA.h5ad
* http://download.gao-lab.org/GLUE/dataset/Muto-2021-FRAGS2RNA.h5ad
* http://download.gao-lab.org/GLUE/dataset/Yao-2021-FRAGS2RNA.h5ad

# Experiments

## Case studies

* TripleOmics: Triple omic integration of scRNA-seq, scATAC-seq and snmC-seq in the mouse cortex
* RegInf: Regulatory inference in PBMC via integrating scRNA-seq, scATAC-seq along with pcHi-C and eQTL
* Atlas: Mega-scale scRNA-seq, scATAC-seq integration of a human cell atlas

## Execution

Go to the subdirectories for each case study, and execute:

```sh
snakemake --use-conda --res mutex=1 -j20 -prk
```

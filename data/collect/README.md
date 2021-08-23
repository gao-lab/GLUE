# Data collection

Jupyter notebooks are used to proprocess and convert downloaded data into standardized formats.

## Single-cell genomics datasets

### RNA

* Data format is ".h5ad".
* Data matrix (`X`) should use raw count whenever available.
* Cell meta (`obs`) should contain cell type annotation in "cell_type" slot whenever available, and additional "domain", "dataset" columns indicating the genomic layer and dataset name.
* Gene meta (`var`) should contain genomic positions following BED specification, and an additional "genome" column indicating the genome assembly version.
* Genes not detected in any cell should be filtered out.
* Highly variable genes should be selected.
* Index name of `obs` and `var` should be named "cells" and "genes".

### ATAC

* Data format is ".h5ad".
* Data matrix (`X`) should use raw count whenever available.
* Cell meta (`obs`) should contain cell type annotation in "cell_type" slot whenever available, and additional "domain", "dataset" columns indicating the genomic layer and dataset name.
* Peak meta (`var`) should contain genomic positions following BED specification, and an additional "genome" column indicating the genome assembly version
* Peaks not detected in any cell should be filtered out.
* Peaks located in ENCODE blacklist regions should be filtered out.
* Index name of `obs` and `var` should be named "cells" and "peaks".

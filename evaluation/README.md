# Method evaluation

## Methods

* UnionCom: Cao, K., Bai, X., Hong, Y. & Wan, L. Unsupervised topological alignment for single-cell multi-omics integration. *Bioinformatics* **36**, i48–i56, [doi:10.1093/bioinformatics/btaa443](https://doi.org/10.1093/bioinformatics/btaa443) (2020).
* LIGER: Welch, J. D. et al. Single-Cell Multi-omic Integration Compares and Contrasts Features of Brain Cell Identity. *Cell* **177**, 1873-1887 e1817, [doi:10.1016/j.cell.2019.05.006](https://doi.org/10.1016/j.cell.2019.05.006) (2019).
* bindSC: Dou, J. et al. Unbiased integration of single cell multi-omics data. *bioRxiv*, [2020.12.11.422014](https://www.biorxiv.org/content/10.1101/2020.12.11.422014) (2020).
* CCA anchor: Stuart, T. et al. Comprehensive Integration of Single-Cell Data. *Cell* **177**, 1888-1902 e1821, [doi:10.1016/j.cell.2019.05.031](https://doi.org/10.1016/j.cell.2019.05.031) (2019).

## Execution

```sh
snakemake -j32 --profile profiles/local  # On local server
snakemake -j500 --profile profiles/cls  # On HPC cluster
```

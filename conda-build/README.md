# Building the conda package

After updating the recipe, run:

```sh
conda build -c defaults -c pytorch -c bioconda -c conda-forge scglue
anaconda upload -u scglue path_to_the_built_package.tar.bz2
```

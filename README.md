# GLUE (Graph-Linked Unified Embedding)

[![stars-badge](https://img.shields.io/github/stars/gao-lab/GLUE?logo=GitHub&color=yellow)](https://github.com/gao-lab/GLUE/stargazers)
[![pypi-badge](https://img.shields.io/pypi/v/scglue)](https://pypi.org/project/scglue)
[![conda-badge](https://anaconda.org/bioconda/scglue/badges/version.svg)](https://anaconda.org/bioconda/scglue)
[![docs-badge](https://readthedocs.org/projects/scglue/badge/?version=latest)](https://scglue.readthedocs.io/en/latest/?badge=latest)
[![build-badge](https://github.com/gao-lab/GLUE/actions/workflows/build.yml/badge.svg)](https://github.com/gao-lab/GLUE/actions/workflows/build.yml)
[![coverage-badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/Jeff1995/e704b2f886ff6a37477311b90fdf7efa/raw/coverage.json)](https://github.com/gao-lab/GLUE/actions/workflows/build.yml)
[![license-badge](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> ### News
>
> Latest update: A Chinese manual is now available at [https://scglue.readthedocs.io/zh_CN/latest/](https://scglue.readthedocs.io/zh_CN/latest/)!
>
> 最新更新: GLUE的官方中文手册已上线，请见[https://scglue.readthedocs.io/zh_CN/latest/](https://scglue.readthedocs.io/zh_CN/latest/)！

Graph-linked unified embedding for single-cell multi-omics data integration

![Model architecture](docs/_static/architecture.svg)

For more details, please check out our [publication](https://doi.org/10.1038/s41587-022-01284-4).

## Directory structure

```
.
├── scglue                  # Main Python package
├── data                    # Data files
├── evaluation              # Method evaluation pipelines
├── experiments             # Experiments and case studies
├── tests                   # Unit tests for the Python package
├── docs                    # Documentation files
├── custom                  # Customized third-party packages
├── packrat                 # Reproducible R environment via packrat
├── env.yaml                # Reproducible Python environment via conda
├── pyproject.toml          # Python package metadata
├── LICENSE
└── README.md
```

## Installation

The `scglue` package can be installed via conda using one of the following commands:

```sh
conda install -c conda-forge -c bioconda scglue  # CPU only
conda install -c conda-forge -c bioconda scglue pytorch-gpu  # With GPU support
```

Or, it can also be installed via pip:

```sh
pip install scglue
```

> Installing within a
> [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
> is recommended.

## Usage

Please checkout the documentations and tutorials at
[scglue.readthedocs.io](https://scglue.readthedocs.io).

A Chinese version is also available [here](https://scglue.readthedocs.io/zh_CN/latest/).

## Development

Install scglue in editable form via flit (first install flit via conda or pip
if not installed already):

```sh
flit install -s
```

Run unit tests:

```sh
pytest --cov="scglue" --cov-report="term-missing" tests [--cpu-only]
```

Build documentation:

```sh
sphinx-build -b gettext docs docs/_build/gettext
sphinx-intl update -p docs/_build/gettext -l zh_CN -d docs/locale
sphinx-build -b html -D language=en docs docs/_build/html/en  # English version
sphinx-build -b html -D language=zh_CN docs docs/_build/html/zh_CN  # Chinese version
```

## Reproduce results

1. Checkout the repository to v0.2.0:

   ```sh
   git checkout tags/v0.2.0
   ```

2. Create a local conda environment using the `env.yaml` file,
and then install scglue:

   ```sh
   conda env create -p conda -f env.yaml && conda activate ./conda
   flit install -s
   ```

3. Set up a project-specific R environment:

   ```R
   packrat::restore()  # Packrat should be automatically installed if not available.
   install.packages("data/download/Saunders-2018/DropSeq.util_2.0.tar.gz", repos = NULL)
   install.packages("custom/Seurat_4.0.2.tar.gz", lib = "packrat/custom", repos = NULL)
   ```

   > R 4.0.2 was used during the project, but any version above 4.0.0 should be compatible.

4. Follow instructions in `data` to prepare the necessary data.
5. Follow instructions in `evaluation` for method evaluation.
6. Follow instructions in `experiments` for case studies.

Installation guide
==================

************
Main package
************

The ``scglue`` package can be installed via conda using one of the following commands:

.. code-block:: bash
    :linenos:

    conda install -c conda-forge -c bioconda scglue  # CPU only
    conda install -c conda-forge -c bioconda scglue pytorch-gpu  # With GPU support

Or, it can also be installed via pip:

.. code-block:: bash
    :linenos:

    pip install scglue

.. note::
    To avoid potential dependency conflicts, installing within a
    `conda environment <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`__
    is recommended.


*********************
Optional dependencies
*********************

Some functions in the ``scglue`` package use metacell aggregation via k-Means clustering,
which can receive significant speed up with the `faiss <https://github.com/facebookresearch/faiss>`__ package.

You may install ``faiss`` following the official `guide <https://github.com/facebookresearch/faiss/blob/main/INSTALL.md>`__.

Functions that use Leiden clustering, i.e., `estimate_balancing_weight <api/scglue.data.estimate_balancing_weight.rst>`__
and `normalized_mutual_info <api/scglue.metrics.normalized_mutual_info.rst>`__,
require the `leidenalg <https://github.com/vtraag/leidenalg>`__ package.
It is not installed by default and can be installed via:

.. code-block:: bash
    :linenos:

    pip install scglue[leiden]

Now you are all set. Proceed to `tutorials <tutorials.rst>`__ for how to use the ``scglue`` package.

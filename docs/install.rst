Installation guide
==================

************
Main package
************

The ``scglue`` package can be installed either via conda:

.. code-block:: bash
    :linenos:

    conda install -c defaults -c pytorch -c bioconda -c conda-forge -c scglue scglue

Or via pip:

.. code-block:: bash
    :linenos:

    pip install scglue

.. note::
    To avoid potential dependency conflicts, installing within a
    `conda environment <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
    is recommended.


*********************
Optional dependencies
*********************

Some functions in the ``scglue`` package use metacell aggregation via k-Means clustering,
which can receive significant speed up with the `faiss <https://github.com/facebookresearch/faiss>`_ package.

You may install ``faiss`` following the official `guide <https://github.com/facebookresearch/faiss/blob/main/INSTALL.md>`_.

Now you are all set. Proceed to `tutorials <tutorials.rst>`_ for how to use the ``scglue`` package.

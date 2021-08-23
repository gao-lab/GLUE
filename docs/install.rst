Installation guide
==================

************
Main package
************

It is recommended to install ``scglue`` in a `conda <https://docs.conda.io/en/latest/miniconda.html>`_ environment, to avoid potential dependency conflicts.

First create a new conda environment (here named "scglue", but you may name it whatever you like), and activate it:

.. code-block:: bash
    :linenos:

    conda create -n scglue python=3.8
    conda activate scglue

Then install ``scglue`` via:

.. code-block:: bash
    :linenos:

    pip install scglue

*********************
Optional dependencies
*********************

In order to use some of the genomics-related functions, it is necessary to have `bedtools <https://bedtools.readthedocs.io/en/latest/index.html>`_ installed (at least v2.29.2). You may also install it from conda:

.. code-block:: bash
    :linenos:

    conda install -c bioconda bedtools>=2.29.2

.. note:: A conda package is also being prepared, which would be more convenient. Stay tuned.

Now you are all set. Proceed to `tutorials <tutorials.rst>`_ for how to use the ``scglue`` package.

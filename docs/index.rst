Welcome to scglue's documentation!
==================================

.. image:: _static/icon.svg
   :width: 140
   :alt: GLUE icon
   :align: right

``scglue`` implements the GLUE (**G**\ raph **L**\ inked **U**\ nified **E**\ mbedding) model, for unpaired single-cell multi-omics data integration.

GLUE is a flexible framework that utilizes prior knowledge about feature relations to bridge the gap between different feature spaces during unpaired multi-modal data integration.

In the context of single-cell multi-omics, the data modalities correspond to omics layers, e.g., scRNA-seq, scATAC-seq, snmC-seq, etc. "Unpaired" means that different omics layers are not probed in the same single cells, but rather independent samples of (presumably) the same cell population. Prior knowledge consists of prior regulatory interactions between omics features, e.g., RNA genes and ATAC peaks in the case of scRNA-seq and scATAC-seq integration. These interactions are compiled into a guidance graph, and utilized by GLUE to help orient the multi-omics integration.

.. image:: _static/architecture.svg
   :width: 600
   :alt: Model architecture

To get started with ``scglue``, check out the `installation guide <install.rst>`__ and `tutorials <tutorials.rst>`__.

For more details about the GLUE framework, please check out our `publication <https://doi.org/10.1038/s41587-022-01284-4>`__.

For a Chinese version of this documentation, please visit `https://scglue.readthedocs.io/zh_CN/latest/ <https://scglue.readthedocs.io/zh_CN/latest/>`__.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   tutorials
   data
   api
   release
   dev
   credits

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`

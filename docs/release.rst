Release notes
=============

v0.2.1
------

Minor improvements and dependency fixes


v0.2.0
------

New features:

- Added `fit_SCGLUE <api/scglue.models.fit_SCGLUE.rst>`_ function to simplify model training
  - Incorporates weighted adversarial alignment by default, with increased robustness on datasets with highly-skewed cell type compositions
- Added support for batch effect correction, which can be activated by setting ``use_batch`` in `configure_dataset <api/scglue.models.scglue.configure_dataset.rst>`_
- Added a model diagnostics metric `"integration consistency score" <api/scglue.models.dx.integration_consistency.rst>`_

Enhancements:

- Support for training directly on disk-backed AnnData objects, scaling to almost infinite number of cells

Bug fixes:

- Fixed a bug where the graph dataset was not shuffled across epochs

Experimental features:

- A `partially paired GLUE model <api/scglue.models.scglue.PairedSCGLUEModel.rst>`_ for utilizing paired cells whenever available
- The `CLUE model <api/scglue.models.scclue.SCCLUEModel.rst>`_ that won the `NeurIPS 2020 competition in multimodal integration <https://openproblems.bio/neurips_2021/>`_ is here!


v0.1.1
------

First public release

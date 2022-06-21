Release notes
=============

v0.2.3
------

Minor improvements and bug fixes

Bug fixes:

- Data frame in ``obsm`` no longer triggers an error during model training (Resolves `#32 <https://github.com/gao-lab/GLUE/issues/32>`_).

Enhancements:

- `scglue.data.transfer_labels <api/scglue.data.transfer_labels.rst>`_ uses a new strategy with SNN-based estimation of transfer confidence (Resolves `#23 <https://github.com/gao-lab/GLUE/issues/23>`_).
- Allow setting custom bedtools path via `scglue.config.BEDTOOLS_PATH <api/scglue.utils.ConfigManager.rst>`_ (Resolves `#22 <https://github.com/gao-lab/GLUE/issues/22>`_).

v0.2.2
------

Minor improvements and bug fixes

Bug fixes:

- Device detection is now more reliable (Resolves `#17 <https://github.com/gao-lab/GLUE/issues/17>`_).

Enhancements:

- Custom encoders and decoders can now be registered without changing package code (Resolves `#14 <https://github.com/gao-lab/GLUE/issues/14>`_).


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
- The `CLUE model <api/scglue.models.scclue.SCCLUEModel.rst>`_ that won the `NeurIPS 2021 competition in multimodal integration <https://openproblems.bio/neurips_2021/>`_ is here!


v0.1.1
------

First public release

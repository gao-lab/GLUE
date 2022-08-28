Release notes
=============

v0.3.2
------

Bug fixes:

- Fixed "real_cross" loss in `PairedSCGLUETrainer <api/scglue.models.scglue.PairedSCGLUETrainer.rst>`__ and `SCCLUETrainer <api/scglue.models.scclue.SCCLUETrainer>`__.

v0.3.1
------

Bug fixes:

- Fixed NaN loss in `PairedSCGLUETrainer <api/scglue.models.scglue.PairedSCGLUETrainer.rst>`__.
- Restored `rna_anchored_prior_graph <api/scglue.genomics.rna_anchored_prior_graph.rst>`__ as a deprecated function
  (to be replaced by `rna_anchored_guidance_graph <api/scglue.genomics.rna_anchored_guidance_graph.rst>`__).

v0.3.0
------

New features:

- New `tutorial <reginf.ipynb>`__ and functions for regulatory inference (Resolves `#15 <https://github.com/gao-lab/GLUE/issues/15>`__, `#41 <https://github.com/gao-lab/GLUE/issues/41>`__).
- New `tutorial <paired.ipynb>`__ for training on partially paired data (Resolves `#24 <https://github.com/gao-lab/GLUE/issues/24>`__).

Enhancements:

- Modularized `scglue.models.integration_consistency <api/scglue.models.dx.integration_consistency.rst>`__ to allow for non-raw-count input (Resolves `#30 <https://github.com/gao-lab/GLUE/issues/30>`__).
- Added documentation translation in Chinese.

v0.2.3
------

Minor improvements and bug fixes

Bug fixes:

- Data frame in ``obsm`` no longer triggers an error during model training (Resolves `#32 <https://github.com/gao-lab/GLUE/issues/32>`__).

Enhancements:

- `scglue.data.transfer_labels <api/scglue.data.transfer_labels.rst>`__ uses a new strategy with SNN-based estimation of transfer confidence (Resolves `#23 <https://github.com/gao-lab/GLUE/issues/23>`__).
- Allow setting custom bedtools path via `scglue.config.BEDTOOLS_PATH <api/scglue.utils.ConfigManager.rst>`__ (Resolves `#22 <https://github.com/gao-lab/GLUE/issues/22>`__).

v0.2.2
------

Minor improvements and bug fixes

Bug fixes:

- Device detection is now more reliable (Resolves `#17 <https://github.com/gao-lab/GLUE/issues/17>`__).

Enhancements:

- Custom encoders and decoders can now be registered without changing package code (Resolves `#14 <https://github.com/gao-lab/GLUE/issues/14>`__).


v0.2.1
------

Minor improvements and dependency fixes


v0.2.0
------

New features:

- Added `fit_SCGLUE <api/scglue.models.fit_SCGLUE.rst>`__ function to simplify model training
  - Incorporates weighted adversarial alignment by default, with increased robustness on datasets with highly-skewed cell type compositions
- Added support for batch effect correction, which can be activated by setting ``use_batch`` in `configure_dataset <api/scglue.models.scglue.configure_dataset.rst>`__
- Added a model diagnostics metric `"integration consistency score" <api/scglue.models.dx.integration_consistency.rst>`__

Enhancements:

- Support for training directly on disk-backed AnnData objects, scaling to almost infinite number of cells

Bug fixes:

- Fixed a bug where the graph dataset was not shuffled across epochs

Experimental features:

- A `partially paired GLUE model <api/scglue.models.scglue.PairedSCGLUEModel.rst>`__ for utilizing paired cells whenever available
- The `CLUE model <api/scglue.models.scclue.SCCLUEModel.rst>`__ that won the `NeurIPS 2021 competition in multimodal integration <https://openproblems.bio/neurips_2021/>`__ is here!


v0.1.1
------

First public release

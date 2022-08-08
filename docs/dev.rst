Developer guide
===============

.. note::
    To better understand the following guide, you may check out our
    `publication <https://doi.org/10.1038/s41587-022-01284-4>`__
    first to learn about the general idea.

The GLUE framework is designed to be modular, and can be extended in the many ways.

Below we describe main components of the framework, and how to extend the existing implementations.

***************
Main components
***************

A GLUE model is primarily composed of four main components (all PyTorch `Modules <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`__):

- `Data encoders <api/scglue.models.sc.DataEncoder.rst>`__ (one for each domain)
  - A data encoder receives data input :math:`x`, and returns a distribution corresponding to the data posterior (cell embeddings) :math:`q(u|x)`
- `Data decoders <api/scglue.models.sc.DataDecoder.rst>`__ (one for each domain)
  - A data decoder receives cell embedding input :math:`u` and feature embedding input :math:`v`, and returns a distribution corresponding to the data likelihood :math:`p(x|u, v)`
- A `graph encoder <api/scglue.models.sc.GraphEncoder.rst>`__
  - A graph encoder receives graph input :math:`\mathcal{G}` in the form of edge index, edge weight, and edge sign, and returns a distribution corresponding to the graph posterior (feature embeddings) :math:`q(v|\mathcal{G})`
- A `graph decoder <api/scglue.models.sc.GraphDecoder.rst>`__
  - A graph decoder receives feature embedding input :math:`v`, as well as a subset of query edges in the form of edge index and edge sign, and returns a distribution corresponding to the likelihood of these query edges, which is used as an estimate of the graph likelihood :math:`p(\mathcal{G}|v)`

Current implementations for these components are all located in `scglue.models.sc <api/scglue.models.sc.rst>`__. New extensions can be added to this module as well.

Actual module inferfaces differ slightly from those summarized above, e.g., with additional considerations for library size normalization and batch effect. See below for details.

***************************
Support new data modalities
***************************

A straighforward extension is to add new data encoders and decoders to support additional data modalities.

Define encoder
--------------

Data encoders should inherit from the `DataEncoder <api/scglue.models.sc.DataEncoder.rst>`__ class.
The main part of the encoder is an MLP (Multi-Layer Perceptron) already implemented in `DataEncoder <api/scglue.models.sc.DataEncoder.rst>`__. It leaves two customizable abstract methods:

- The ``compute_l`` method is supposed to compute a library size from the input data
- The ``normalize`` method is supposed to normalize the input data (potentially with the computed library size), before feeding to the MLP.

Below is an example of a negative binomial data encoder, which accepts raw counts as input. The library size is computed simply by summing counts in each cell, while data normlization is performed by row normalizing to a constant size of 10000 and then log-transformed.

.. code-block:: python

  class NBDataEncoder(DataEncoder):

    TOTAL_COUNT = 1e4

    def compute_l(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=1, keepdim=True)

    def normalize(
            self, x: torch.Tensor, l: torch.Tensor
    ) -> torch.Tensor:
        return (x * (self.TOTAL_COUNT / l)).log1p()

You may define your own encoder class by implementing these two methods as appropriate for the data modality.

Define decoder
--------------

Data decoders should inherit from the `DataDecoder <api/scglue.models.sc.DataDecoder.rst>`__ class. It defines the interface of the constructor as well as the abstract ``forward`` method.

The constructor can accept an output dimensionality ``out_features`` and the number of batches ``n_batches`` (batch as in batch effect).
The ``forward`` method accepts four inputs:

- ``u`` is the cell embeddings
- ``v`` is the feature embedddings
- ``b`` is a batch index
- ``l`` is the library size computed by the encoder

and returns the data likelihood distribution.

Below is an example of a negative binomial data decoder.
It includes three trainable parameters ``scale_lin``, ``bias``, and ``log_theta`` (you may define your own parameters as necessary):

- ``scale_lin`` gives the scale parameter :math:`\alpha` after soft-plus transformation to ensure positivity
- ``bias`` is the bias parameter :math:`\beta`
- ``log_theta`` is log of the inverse dispersion parameter :math:`\theta` of negative binomial

All the three parameters are defined as batch-specific (each batch parameterized by a different row).

The mean of negative binomial (``mu``) is computed via scaling and shifting the inner product of cell and feature embeddings, followed by softmax and library size multiplication. The return value is a negative binomial distribution.

.. code-block:: python

  class NBDataDecoder(DataDecoder):

      def __init__(self, out_features: int, n_batches: int = 1) -> None:
          super().__init__(out_features, n_batches=n_batches)
          self.scale_lin = torch.nn.Parameter(torch.zeros(n_batches, out_features))
          self.bias = torch.nn.Parameter(torch.zeros(n_batches, out_features))
          self.log_theta = torch.nn.Parameter(torch.zeros(n_batches, out_features))

      def forward(
              self, u: torch.Tensor, v: torch.Tensor,
              b: torch.Tensor, l: torch.Tensor
      ) -> D.NegativeBinomial:
          scale = F.softplus(self.scale_lin[b])
          logit_mu = scale * (u @ v.t()) + self.bias[b]
          mu = F.softmax(logit_mu, dim=1) * l
          log_theta = self.log_theta[b]
          return D.NegativeBinomial(
              log_theta.exp(),
              logits=(mu + EPS).log() - log_theta
          )

Note how the batch index ``b`` is used as a row indexer into ``scale_lin``, ``bias`` and ``log_theta``.

You may define your own decoder class by implementing the ``forward`` method to produce likelihood distributions appropriate for the data modality.

Non-standard distributions can also be defined in `scglue.models.prob <api/scglue.models.prob.rst>`__.

Register custom encoder and decoder
-----------------------------------

Finally, use the `scglue.models.scglue.register_prob_model <api/scglue.models.scglue.register_prob_model.rst>`__ function to register the the custom encoder and decoder under a new "prob_model", so they can be activated with a matching ``prob_model`` setting in `configure_dataset <api/scglue.models.scglue.configure_dataset.rst>`__.

**************************
Other types of extensions?
**************************

If you are interested in extending the model in other ways, please open an issue on `Github <https://github.com/gao-lab/GLUE>`__.

**************************
Contributions are welcome!
**************************

Be sure to submit a pull request on `Github <https://github.com/gao-lab/GLUE>`__
if you want your extension to be included in the framework!

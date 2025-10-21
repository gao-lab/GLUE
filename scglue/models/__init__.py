r"""
Integration models
"""

import os
from pathlib import Path
from typing import Mapping, Optional

import dill
import networkx as nx
import numpy as np
import pandas as pd
from anndata import AnnData

from ..data import configure_dataset, estimate_balancing_weight, get_dataset_config
from ..typehint import Kws
from ..utils import config, logged
from .base import Model
from .dx import integration_consistency
from .nn import autodevice
from .scclue import SCCLUEModel
from .scglue import PairedSCGLUEModel, SCGLUEModel


def load_model(fname: os.PathLike) -> Model:
    r"""
    Load model from file

    Parameters
    ----------
    fname
        Specifies path to the file

    Returns
    -------
    model
        Loaded model
    """
    fname = Path(fname)
    with fname.open("rb") as f:
        model = dill.load(f)
    model.upgrade()  # pylint: disable=no-member
    model.net.device = autodevice()  # pylint: disable=no-member
    return model


@logged
def fit_SCGLUE(
    adatas: Mapping[str, AnnData],
    graph: nx.Graph,
    model: type = SCGLUEModel,
    skip_balance: bool = False,
    init_kws: Kws = None,
    compile_kws: Kws = None,
    fit_kws: Kws = None,
    balance_kws: Kws = None,
) -> SCGLUEModel:
    r"""
    Fit GLUE model to integrate single-cell multi-omics data

    Parameters
    ----------
    adatas
        Single-cell datasets (indexed by modality name)
    graph
        Guidance graph
    model
        Model class, must be one of
        {:class:`scglue.models.scglue.SCGLUEModel`,
        :class:`scglue.models.scglue.PairedSCGLUEModel`}
    skip_balance
        Skip the balancing step and use equal weight for all cells
    init_kws
        Model initialization keyword arguments
        (see the constructor of the ``model`` class,
        either :class:`scglue.models.scglue.SCGLUEModel`
        or :class:`scglue.models.scglue.PairedSCGLUEModel`)
    compile_kws
        Model compile keyword arguments
        (see the ``compile`` method of the ``model`` class,
        either :meth:`scglue.models.scglue.SCGLUEModel.compile`
        or :meth:`scglue.models.scglue.PairedSCGLUEModel.compile`)
    fit_kws
        Model fitting keyword arguments
        (see :meth:`scglue.models.scglue.SCGLUEModel.fit`)
    balance_kws
        Balancing weight estimation keyword arguments
        (see :func:`scglue.data.estimate_balancing_weight`)

    Returns
    -------
    model
        Fitted model object
    """
    init_kws = init_kws or {}
    compile_kws = compile_kws or {}
    fit_kws = fit_kws or {}
    balance_kws = balance_kws or {}

    fit_SCGLUE.logger.info("Pretraining SCGLUE model...")
    pretrain_init_kws = init_kws.copy()
    pretrain_init_kws.update({"shared_batches": False})
    pretrain_fit_kws = fit_kws.copy()
    pretrain_fit_kws.update({"align_burnin": np.inf, "safe_burnin": False})
    if "directory" in pretrain_fit_kws:
        pretrain_fit_kws["directory"] = os.path.join(
            pretrain_fit_kws["directory"], "pretrain"
        )

    pretrain = model(adatas, sorted(graph.nodes), **pretrain_init_kws)
    pretrain.compile(**compile_kws)
    pretrain.fit(adatas, graph, **pretrain_fit_kws)
    if "directory" in pretrain_fit_kws:
        pretrain.save(os.path.join(pretrain_fit_kws["directory"], "pretrain.dill"))

    if not skip_balance:
        fit_SCGLUE.logger.info("Estimating balancing weight...")
        for k, adata in adatas.items():
            adata.obsm[f"X_{config.TMP_PREFIX}"] = pretrain.encode_data(k, adata)
        if init_kws.get("shared_batches"):
            use_batch = set(
                get_dataset_config(adata)["use_batch"] for adata in adatas.values()
            )
            use_batch = use_batch.pop() if len(use_batch) == 1 else None
        else:
            use_batch = None
        estimate_balancing_weight(
            *adatas.values(),
            use_rep=f"X_{config.TMP_PREFIX}",
            use_batch=use_batch,
            key_added="balancing_weight",
            **balance_kws,
        )
        for adata in adatas.values():
            get_dataset_config(adata)["use_dsc_weight"] = "balancing_weight"
            del adata.obsm[f"X_{config.TMP_PREFIX}"]

    fit_SCGLUE.logger.info("Fine-tuning SCGLUE model...")
    finetune_fit_kws = fit_kws.copy()
    if "directory" in finetune_fit_kws:
        finetune_fit_kws["directory"] = os.path.join(
            finetune_fit_kws["directory"], "fine-tune"
        )

    finetune = model(adatas, sorted(graph.nodes), **init_kws)
    finetune.adopt_pretrained_model(pretrain)
    finetune.compile(**compile_kws)
    fit_SCGLUE.logger.debug(
        "Increasing random seed by 1 to prevent identical data order..."
    )
    finetune.random_seed += 1
    finetune.fit(adatas, graph, **finetune_fit_kws)
    if "directory" in finetune_fit_kws:
        finetune.save(os.path.join(finetune_fit_kws["directory"], "fine-tune.dill"))

    return finetune

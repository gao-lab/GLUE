r"""
Graph-linked unified embedding (GLUE) for single-cell multi-omics
data integration
"""

import os
from itertools import chain
from math import ceil
from typing import List, Mapping, Optional, Tuple

import anndata
import ignite
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.distributions as D
import torch.nn.functional as F

from ..num import normalize_edges
from ..typehint import Array
from ..utils import config, logged
from . import glue, nn, sc
from .base import Model


AUTO = -1  # Flag for using automatically determined hyperparameters


#---------------------------------- Utilities ----------------------------------

def select_encoder(prob_model: str) -> type:
    r"""
    Select encoder architecture

    Parameters
    ----------
    prob_model
        Data probabilistic model

    Return
    ------
    encoder
        Encoder type
    """
    if prob_model in ("Normal", "ZIN", "ZILN"):
        return sc.VanillaDataEncoder
    if prob_model in ("NB", "ZINB"):
        return sc.NBDataEncoder
    raise ValueError("Invalid `prob_model`!")


def select_decoder(prob_model: str) -> type:
    r"""
    Select decoder architecture

    Parameters
    ----------
    prob_model
        Data probabilistic model

    Return
    ------
    decoder
        Decoder type
    """
    if prob_model == "Normal":
        return sc.NormalDataDecoder
    if prob_model == "ZIN":
        return sc.ZINDataDecoder
    if prob_model == "ZILN":
        return sc.ZILNDataDecoder
    if prob_model == "NB":
        return sc.NBDataDecoder
    if prob_model == "ZINB":
        return sc.ZINBDataDecoder
    raise ValueError("Invalid `prob_model`!")


@logged
class DomainConfig:

    r"""
    Domain configuration

    Parameters
    ----------
    prob_model
        Data probabilistic model
    features
        List of domain feature names
    use_layer
        Data layer to use (key in ``adata.layers``)
    use_rep
        Data representation to use as the first encoder transformation
        (key in ``adata.obsm``)
    rep_dim
        Data representation dimensionality
    use_dsc_weight
        Discriminator sample weight to use (key in ``adata.obs``)
    **kwargs
        Additional keyword arguments are ignored
    """

    def __init__(
            self, prob_model: str = None,
            features: List[str] = None,
            use_layer: Optional[str] = None,
            use_rep: Optional[str] = None,
            rep_dim: Optional[int] = None,
            use_dsc_weight: Optional[str] = None,
            **kwargs  # pylint: disable=unused-argument
    ) -> None:
        self.prob_model = prob_model
        self.features = features
        self.use_layer = use_layer
        self.use_rep = use_rep
        self.rep_dim = rep_dim
        self.use_dsc_weight = use_dsc_weight

    def extract_data(
            self, adata: anndata.AnnData, mode: str = "train"
    ) -> Tuple[Array, Array]:
        r"""
        Extract data according to the domain configuration

        Parameters
        ----------
        anndata
            Input dataset
        mode
            Extraction mode, must be one of ``{"train", "eval"}``.

        Returns
        -------
        x
            Data array
        xalt
            Alternative input array
        xdwt
            Discriminator sample weight
        """
        default_dtype = nn.get_default_numpy_dtype()
        empty = np.empty((adata.shape[0], 0), dtype=default_dtype)
        if self.use_rep:
            if self.use_rep not in adata.obsm:
                raise ValueError(
                    f"Configured data representation '{self.use_rep}' "
                    f"cannot be found in input data!"
                )
            xalt = adata.obsm[self.use_rep].astype(default_dtype)
            if xalt.shape[1] != self.rep_dim:
                raise ValueError(
                    f"Input data representation dimensionality {xalt.shape[1]} "
                    f"does not match the configured dimensionality {self.rep_dim}!"
                )
            if mode == "eval":
                return empty, xalt, empty
        else:
            xalt = empty
        adata = adata[:, self.features]
        if self.use_layer:
            if self.use_layer not in adata.layers:
                raise ValueError(
                    f"Configured data layer '{self.use_layer}' "
                    f"cannot be found in input data!"
                )
            x = adata.layers[self.use_layer].astype(default_dtype)
        else:
            x = adata.X.astype(default_dtype)
        if mode == "eval":
            return x, xalt, empty
        if self.use_dsc_weight:
            if self.use_dsc_weight not in adata.obs:
                raise ValueError(
                    f"Configured discriminator sample weight '{self.use_dsc_weight}' "
                    f"cannot be found in input data!"
                )
            xdwt = adata.obs[self.use_dsc_weight].to_numpy().astype(default_dtype)
        else:
            xdwt = np.ones(adata.shape[0], dtype=default_dtype)
        return x, xalt, xdwt


#----------------------------- Network definition ------------------------------

class SCGLUE(glue.GLUE):

    r"""
    GLUE network for single-cell multi-omics data integration

    Parameters
    ----------
    g2v
        Graph encoder
    v2g
        Graph decoder
    x2u
        Data encoders (indexed by domain name)
    u2x
        Data decoders (indexed by domain name)
    idx
        Feature indices among graph vertices (indexed by domain name)
    du
        Domain discriminator
    prior
        Latent prior
    """

    def __init__(
            self, g2v: sc.GraphEncoder, v2g: sc.GraphDecoder,
            x2u: Mapping[str, sc.DataEncoder],
            u2x: Mapping[str, sc.DataDecoder],
            idx: Mapping[str, torch.Tensor],
            du: sc.Discriminator, prior: sc.Prior,
    ) -> None:
        super().__init__(g2v, v2g, x2u, u2x, idx, du, prior)


#----------------------------- Trainer definition ------------------------------

DataTensors = Tuple[
    Mapping[str, torch.Tensor],  # x (data)
    Mapping[str, torch.Tensor],  # xalt (alternative input data)
    Mapping[str, torch.Tensor],  # xdwt (domain discriminator sample weight)
    Mapping[str, torch.Tensor],  # xflag (domain indicator)
    torch.Tensor,  # eidx (edge index)
    torch.Tensor,  # ewt (edge weight)
    torch.Tensor  # esgn (edge sign)
]  # Specifies the data format of input to SCGLUETrainer.compute_losses


@logged
class SCGLUETrainer(glue.GLUETrainer):

    r"""
    Trainer for :class:`SCGLUE`

    Parameters
    ----------
    net
        :class:`SCGLUE` network to be trained
    lam_graph
        Graph weight
    lam_align
        Adversarial alignment weight
    optim
        Optimizer
    lr
        Learning rate
    **kwargs
        Additional keyword arguments are passed to the optimizer constructor
    """

    def __init__(
            self, net: SCGLUE, lam_graph: float = None, lam_align: float = None,
            optim: str = None, lr: float = None, **kwargs
    ) -> None:
        super().__init__(
            net, lam_graph=lam_graph, lam_align=lam_align,
            optim=optim, lr=lr, **kwargs
        )
        self.freeze_u = False

    @property
    def freeze_u(self) -> bool:
        r"""
        Whether to freeze cell embeddings
        """
        return self._freeze_u

    @freeze_u.setter
    def freeze_u(self, freeze_u: bool) -> None:
        self._freeze_u = freeze_u
        for item in chain(self.net.x2u.parameters(), self.net.du.parameters()):
            item.requires_grad_(not self._freeze_u)

    def compute_losses(
            self, data: DataTensors, epoch: int, dsc_only: bool = False
    ) -> Mapping[str, torch.Tensor]:
        net = self.net
        x, xalt, xdwt, xflag, eidx, ewt, esgn = data

        u, l = {}, {}
        for k in net.keys:
            u[k], l[k] = net.x2u[k](x[k], xalt[k], lazy_normalizer=dsc_only)
        usamp = {k: u[k].rsample() for k in net.keys}
        prior = net.prior()

        u_cat = torch.cat([u[k].mean for k in net.keys])
        xdwt_cat = torch.cat([xdwt[k] for k in net.keys])
        xflag_cat = torch.cat([xflag[k] for k in net.keys])
        anneal = max(1 - (epoch - 1) / self.align_burnin, 0) \
            if self.align_burnin else 0
        if anneal:
            noise = prior.sample(u_cat.shape)
            u_cat = u_cat + anneal * noise
        dsc_loss = F.cross_entropy(net.du(u_cat), xflag_cat, reduction="none")
        dsc_loss = (dsc_loss * xdwt_cat).sum() / xdwt_cat.numel()
        if dsc_only:
            return {"dsc_loss": self.lam_align * dsc_loss}

        v = net.g2v(self.eidx, self.enorm, self.esgn)
        vsamp = v.rsample()

        g_nll = -net.v2g(vsamp, eidx, esgn).log_prob(ewt)
        pos_mask = (ewt != 0).to(torch.int64)
        n_pos = pos_mask.sum().item()
        n_neg = pos_mask.numel() - n_pos
        g_nll_pn = torch.zeros(2, dtype=g_nll.dtype, device=g_nll.device)
        g_nll_pn.scatter_add_(0, pos_mask, g_nll)
        avgc = (n_pos > 0) + (n_neg > 0)
        g_nll = (g_nll_pn[0] / max(n_neg, 1) + g_nll_pn[1] / max(n_pos, 1)) / avgc
        g_kl = D.kl_divergence(v, prior).sum(dim=1).mean() / vsamp.shape[0]
        g_elbo = g_nll + g_kl

        x_nll = {
            k: -net.u2x[k](
                usamp[k], vsamp[getattr(net, f"{k}_idx")], l[k]
            ).log_prob(x[k]).mean()
            for k in net.keys
        }
        x_kl = {
            k: D.kl_divergence(
                u[k], prior
            ).sum(dim=1).mean() / x[k].shape[1]
            for k in net.keys
        }
        x_elbo = {
            k: x_nll[k] + x_kl[k]
            for k in net.keys
        }

        gen_loss = sum(x_elbo.values()) \
            + self.lam_graph * len(net.keys) * g_elbo \
            - self.lam_align * dsc_loss

        losses = {"g_nll": g_nll, "g_kl": g_kl, "g_elbo": g_elbo}
        for k in net.keys:
            losses.update({
                f"x_{k}_nll": x_nll[k],
                f"x_{k}_kl": x_kl[k],
                f"x_{k}_elbo": x_elbo[k]
            })
        losses.update({"dsc_loss": dsc_loss, "gen_loss": gen_loss})
        return losses

    def format_data(self, data: List[torch.Tensor]) -> DataTensors:
        r"""
        Format data tensors

        Note
        ----
        The data dataset should contain data arrays for each domain,
        followed by alternative input arrays for each domain,
        in the same order as domain keys of the network.
        """
        device = self.net.device
        keys = self.net.keys
        K = len(keys)
        x, xalt, xdwt, (eidx, ewt, esgn) = \
            data[:K], data[K:2*K], data[2*K:3*K], data[3*K:]
        x = {
            k: x[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xalt = {
            k: xalt[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xdwt = {
            k: xdwt[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xflag = {
            k: torch.as_tensor(
                i, dtype=torch.int64, device=device
            ).expand(x[k].shape[0])
            for i, k in enumerate(keys)
        }
        eidx = eidx.to(device, non_blocking=True)
        ewt = ewt.to(device, non_blocking=True)
        esgn = esgn.to(device, non_blocking=True)
        return x, xalt, xdwt, xflag, eidx, ewt, esgn

    def train_step(
            self, engine: ignite.engine.Engine, data: List[torch.Tensor]
    ) -> Mapping[str, torch.Tensor]:
        self.net.train()
        data = self.format_data(data)
        epoch = engine.state.epoch

        if self.freeze_u:
            self.net.x2u.apply(nn.freeze_running_stats)
            self.net.du.apply(nn.freeze_running_stats)
        else:  # Discriminator step
            losses = self.compute_losses(data, epoch, dsc_only=True)
            self.net.zero_grad(set_to_none=True)
            losses["dsc_loss"].backward()  # Already scaled by lam_align
            self.dsc_optim.step()

        # Generator step
        losses = self.compute_losses(data, epoch)
        self.net.zero_grad(set_to_none=True)
        losses["gen_loss"].backward()
        self.vae_optim.step()

        return losses

    def __repr__(self):
        vae_optim = repr(self.vae_optim).replace("    ", "  ").replace("\n", "\n  ")
        dsc_optim = repr(self.dsc_optim).replace("    ", "  ").replace("\n", "\n  ")
        return (
            f"{type(self).__name__}(\n"
            f"  lam_graph: {self.lam_graph}\n"
            f"  lam_align: {self.lam_align}\n"
            f"  vae_optim: {vae_optim}\n"
            f"  dsc_optim: {dsc_optim}\n"
            f"  freeze_u: {self.freeze_u}\n"
            f")"
        )


#--------------------------------- Public API ----------------------------------

@logged
def configure_dataset(
        adata: anndata.AnnData, prob_model: str,
        use_highly_variable: bool = True,
        use_layer: Optional[str] = None,
        use_rep: Optional[str] = None,
        use_dsc_weight: Optional[str] = None
) -> None:
    r"""
    Configure dataset for model training

    Parameters
    ----------
    adata
        Dataset to be configured
    prob_model
        Probabilistic generative model used by the decoder,
        must be one of ``{"Normal", "ZIN", "ZILN", "NB", "ZINB"}``.
    use_highly_variable
        Whether to use highly variable features
    use_layer
        Data layer to use (key in ``adata.layers``)
    use_rep
        Data representation to use as the first encoder transformation
        (key in ``adata.obsm``)
    use_dsc_weight
        Discriminator sample weight to use (key in ``adata.obs``)

    Note
    -----
    The ``use_rep`` option applies to encoder inputs, but not the decoders,
    which are always fitted on data in the original space.
    """
    if config.ANNDATA_KEY in adata.uns:
        configure_dataset.logger.warning(
            "`configure_dataset` has already been called. "
            "Previous configuration will be overwritten!"
        )
    data_config = {}
    data_config["prob_model"] = prob_model
    if use_highly_variable:
        if "highly_variable" not in adata.var:
            raise ValueError("Please mark highly variable features first!")
        data_config["use_highly_variable"] = True
        data_config["features"] = adata.var.query("highly_variable").index.to_numpy().tolist()
    else:
        data_config["use_highly_variable"] = False
        data_config["features"] = adata.var_names.to_numpy().tolist()
    if use_layer:
        if use_layer not in adata.layers:
            raise ValueError("Invalid `use_layer`!")
        data_config["use_layer"] = use_layer
    else:
        data_config["use_layer"] = None
    if use_rep:
        if use_rep not in adata.obsm:
            raise ValueError("Invalid `use_rep`!")
        data_config["use_rep"] = use_rep
        data_config["rep_dim"] = adata.obsm[use_rep].shape[1]
    else:
        data_config["use_rep"] = None
        data_config["rep_dim"] = None
    if use_dsc_weight:
        if use_dsc_weight not in adata.obs:
            raise ValueError("Invalid `use_dsc_weight`!")
        data_config["use_dsc_weight"] = use_dsc_weight
    else:
        data_config["use_dsc_weight"] = None
    adata.uns[config.ANNDATA_KEY] = data_config


@logged
class SCGLUEModel(Model):

    r"""
    GLUE model for single-cell multi-omics data integration

    Parameters
    ----------
    adatas
        Datasets (indexed by domain name)
    vertices
        Prior graph vertices (must cover feature names in all domains)
    latent_dim
        Latent dimensionality
    h_depth
        Hidden layer depth for encoder and discriminator
    h_dim
        Hidden layer dimensionality for encoder and discriminator
    dropout
        Dropout rate
    random_seed
        Random seed
    """

    NET_TYPE = SCGLUE
    TRAINER_TYPE = SCGLUETrainer

    GRAPH_BATCHES: int = 32  # Number of graph batches in each graph epoch
    ALIGN_BURNIN_PRG: float = 8.0  # Effective optimization progress of align_burnin (learning rate * iterations)
    MAX_EPOCHS_PRG: float = 32.0  # Effective optimization progress of max_epochs (learning rate * iterations)
    PATIENCE_PRG: float = 4.0  # Effective optimization progress of patience (learning rate * iterations)
    REDUCE_LR_PATIENCE_PRG: float = 2.0  # Effective optimization progress of reduce_lr_patience (learning rate * iterations)

    def __init__(
            self, adatas: Mapping[str, anndata.AnnData],
            vertices: List[str], latent_dim: int = 50,
            h_depth: int = 2, h_dim: Optional[int] = 256,
            dropout: float = 0.2, random_seed: int = 0
    ) -> None:
        self.vertices = pd.Index(vertices)
        self.random_seed = random_seed
        torch.manual_seed(self.random_seed)

        g2v = sc.GraphEncoder(self.vertices.size, latent_dim)
        v2g = sc.GraphDecoder()
        self.domains, x2u, u2x, idx = {}, {}, {}, {}
        for k, adata in adatas.items():
            if config.ANNDATA_KEY not in adata.uns:
                raise ValueError(
                    f"The '{k}' dataset has not been configured. "
                    f"Please call `configure_dataset` first!"
                )
            domain = DomainConfig(**adata.uns[config.ANNDATA_KEY])
            if domain.rep_dim and domain.rep_dim < latent_dim:
                self.logger.warning(
                    "It is recommended that `use_rep` dimensionality "
                    "be equal or larger than `latent_dim`."
                )
            idx[k] = self.vertices.get_indexer(domain.features).astype(np.int64)
            if idx[k].min() < 0:
                raise ValueError("Not all domain features exist in the graph!")
            idx[k] = torch.as_tensor(idx[k])
            x2u[k] = select_encoder(domain.prob_model)(
                domain.rep_dim or len(domain.features), latent_dim,
                h_depth=h_depth, h_dim=h_dim, dropout=dropout
            )
            u2x[k] = select_decoder(domain.prob_model)(len(domain.features))
            self.domains[k] = domain
        du = sc.Discriminator(
            latent_dim, len(self.domains),
            h_depth=h_depth, h_dim=h_dim, dropout=dropout
        )
        prior = sc.Prior()
        super().__init__(g2v, v2g, x2u, u2x, idx, du, prior)

    def freeze_cells(self) -> None:
        r"""
        Freeze cell embeddings
        """
        self.trainer.freeze_u = True

    def unfreeze_cells(self) -> None:
        r"""
        Unfreeze cell embeddings
        """
        self.trainer.freeze_u = False

    def adopt_pretrained_model(
            self, source: "SCGLUEModel", submodule: Optional[str] = None
    ) -> None:
        r"""
        Adopt buffers and parameters from a pretrained model

        Parameters
        ----------
        source
            Source model to be adopted
        submodule
            Only adopt a specific submodule (e.g., ``"x2u"``)
        """
        s, t = source.net, self.net
        if submodule:
            for item in submodule.split("."):
                s = getattr(s, submodule)
                t = getattr(t, submodule)
        for key, val in chain(t.named_parameters(), t.named_buffers()):
            ptr = s
            for item in key.split("."):
                if not hasattr(ptr, item):
                    self.logger.warning("Missing: %s", key)
                    break
                ptr = getattr(ptr, item)
            else:
                if val.shape != ptr.shape:
                    self.logger.warning("Shape mismatch: %s", key)
                else:
                    if isinstance(val, torch.nn.Parameter):
                        val.data.copy_(ptr.data)
                        self.logger.info("Copied parameter: %s", key)
                    else:
                        val.copy_(ptr)
                        self.logger.info("Copied buffer: %s", key)

    def graph2triplet(
            self, graph: nx.Graph, weight: str, sign: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Convert graph object to graph triplet

        Parameters
        ----------
        graph
            Graph object
        weight
            Key of edge attribute for edge weight
        sign
            Key of edge attribute for edge sign

        Returns
        -------
        eidx
            Vertex indices of edges (:math:`2 \times n_{edges}`)
        ewt
            Weight of edges (:math:`n_{edges}`)
        esgn
            Sign of edges (:math:`n_{edges}`)
        """
        if not all(graph.has_edge(v, v) for v in graph.nodes):
            self.logger.warning(
                "Not all vertices contain self-loops! "
                "Self-loops are recommended."
            )
        graph = nx.MultiDiGraph(graph)  # Convert undirecitonal to bidirectional, while keeping multi-edges

        default_dtype = nn.get_default_numpy_dtype()
        i, j, w, s = [], [], [], []
        for k, v in dict(graph.edges).items():
            i.append(k[0])
            j.append(k[1])
            w.append(v[weight])
            s.append(v[sign])
        eidx = np.stack([
            self.vertices.get_indexer(i),
            self.vertices.get_indexer(j)
        ]).astype(np.int64)
        ewt = np.asarray(w).astype(default_dtype)
        esgn = np.asarray(s).astype(default_dtype)
        return eidx, ewt, esgn

    def compile(  # pylint: disable=arguments-differ
            self, lam_graph: float = 0.02, lam_align: float = 0.02,
            lr: float = 2e-3
    ) -> None:
        r"""
        Prepare model for training

        Parameters
        ----------
        lam_graph
            Graph weight
        lam_align
            Adversarial alignment weight
        lr
            Learning rate
        """
        super().compile(
            lam_graph=lam_graph, lam_align=lam_align,
            optim="RMSprop", lr=lr
        )

    def fit(  # pylint: disable=arguments-differ
            self, adatas: Mapping[str, anndata.AnnData], graph: nx.Graph,
            edge_weight: str = "weight", edge_sign: str = "sign",
            neg_samples: int = 10, val_split: float = 0.1,
            data_batch_size: int = 128, graph_batch_size: int = AUTO,
            align_burnin: int = AUTO, max_epochs: int = AUTO,
            patience: Optional[int] = AUTO, reduce_lr_patience: Optional[int] = AUTO,
            directory: Optional[os.PathLike] = None
    ) -> None:
        r"""
        Fit model on given datasets

        Parameters
        ----------
        adatas
            Datasets (indexed by domain name)
        graph
            Prior graph
        edge_weight
            Key of edge attribute for edge weight
        edge_sign
            Key of edge attribute for edge sign
        neg_samples
            Number of negative samples for each edge
        val_split
            Validation split
        data_batch_size
            Number of cells in each data minibatch
        graph_batch_size
            Number of edges in each graph minibatch
        align_burnin
            Number of epochs to wait before starting alignment
        max_epochs
            Maximal number of epochs
        patience
            Patience of early stopping
        reduce_lr_patience
            Patience to reduce learning rate
        directory
            Directory to store checkpoints and tensorboard logs
        """
        x, xalt, xdwt = [], [], []
        for k in self.net.keys:
            x_, xalt_, xdwt_ = self.domains[k].extract_data(adatas[k])
            xdwt_ /= xdwt_.sum() / xdwt_.size
            x.append(x_)
            xalt.append(xalt_)
            xdwt.append(xdwt_)
        data = nn.ArrayDataset(*x, *xalt, *xdwt, grouping=self.net.keys * 3)
        graph = nn.GraphDataset(
            *self.graph2triplet(graph, edge_weight, edge_sign),
            neg_samples=neg_samples,
            weighted_sampling=True,
            deemphasize_loops=True
        )

        # Heuristic settings for data size dependent hyperparameters
        batch_per_epoch = data.size * (1 - val_split) / data_batch_size
        if graph_batch_size == AUTO:
            graph_batch_size = ceil(graph.size / self.GRAPH_BATCHES)
            self.logger.info("Setting `graph_batch_size` = %d", graph_batch_size)
        if align_burnin == AUTO:
            align_burnin = max(
                ceil(self.ALIGN_BURNIN_PRG / self.trainer.lr / batch_per_epoch),
                ceil(self.ALIGN_BURNIN_PRG)
            )
            self.logger.info("Setting `align_burnin` = %d", align_burnin)
        if max_epochs == AUTO:
            max_epochs = max(
                ceil(self.MAX_EPOCHS_PRG / self.trainer.lr / batch_per_epoch),
                ceil(self.MAX_EPOCHS_PRG)
            )
            self.logger.info("Setting `max_epochs` = %d", max_epochs)
        if patience == AUTO:
            patience = max(
                ceil(self.PATIENCE_PRG / self.trainer.lr / batch_per_epoch),
                ceil(self.PATIENCE_PRG)
            )
            self.logger.info("Setting `patience` = %d", patience)
        if reduce_lr_patience == AUTO:
            reduce_lr_patience = max(
                ceil(self.REDUCE_LR_PATIENCE_PRG / self.trainer.lr / batch_per_epoch),
                ceil(self.REDUCE_LR_PATIENCE_PRG)
            )
            self.logger.info("Setting `reduce_lr_patience` = %d", reduce_lr_patience)
        if self.trainer.freeze_u:
            self.logger.info("Cell embeddings are frozen")

        super().fit(
            data, graph, val_split=val_split,
            data_batch_size=data_batch_size, graph_batch_size=graph_batch_size,
            align_burnin=align_burnin,
            max_epochs=max_epochs,
            patience=patience,
            reduce_lr_patience=reduce_lr_patience,
            random_seed=self.random_seed,
            directory=directory
        )

    @torch.no_grad()
    def encode_graph(
            self, graph: nx.Graph,
            edge_weight: str = "weight", edge_sign: str = "sign"
    ) -> np.ndarray:
        r"""
        Compute graph vertex (feature) embedding

        Parameters
        ----------
        graph
            Input graph
        edge_weight
            Key of edge attribute for edge weight
        edge_sign
            Key of edge attribute for edge sign

        Returns
        -------
        embedding
            Graph vertex (feature) embedding
        """
        self.net.eval()
        eidx, ewt, esgn = self.graph2triplet(graph, edge_weight, edge_sign)
        enorm = torch.as_tensor(normalize_edges(eidx, ewt), device=self.net.device)
        esgn = torch.as_tensor(esgn, device=self.net.device)
        eidx = torch.as_tensor(eidx, device=self.net.device)
        return self.net.g2v(eidx, enorm, esgn).mean.detach().cpu().numpy()

    @torch.no_grad()
    def encode_data(
            self, key: str, adata: anndata.AnnData, batch_size: int = 128
    ) -> np.ndarray:
        r"""
        Compute data sample (cell) embedding

        Parameters
        ----------
        key
            Domain key
        adata
            Input dataset
        batch_size
            Size of minibatches

        Returns
        -------
        embedding
            Data sample (cell) embedding
        """
        self.net.eval()
        encoder = self.net.x2u[key]
        x, xalt, _ = self.domains[key].extract_data(adata, mode="eval")
        data = nn.ArrayDataset(x, xalt)
        data_loader = nn.DataLoader(
            data, batch_size=batch_size, shuffle=False, drop_last=False
        )
        result = []
        for x, xalt in data_loader:
            result.append(encoder(
                x.to(self.net.device), xalt.to(self.net.device),
                lazy_normalizer=True
            )[0].mean.detach().cpu())
        return torch.cat(result).numpy()

    def __repr__(self) -> str:
        return (
            f"SCGLUE model with the following network and trainer:\n\n"
            f"{repr(self.net)}\n\n"
            f"{repr(self.trainer)}\n"
        )

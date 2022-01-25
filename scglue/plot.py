r"""
Plotting functions
"""

from typing import Callable, List, Optional, Union

import matplotlib.axes as ma
import numpy as np
import pandas as pd
import seaborn as sns
import scanpy as sc
import sklearn.metrics
from matplotlib import rcParams

from .check import check_deps


#---------------------------- Global configuration -----------------------------

def set_publication_params() -> None:
    r"""
    Set publication-level figure parameters
    """
    sc.set_figure_params(
        scanpy=True, dpi_save=600, vector_friendly=True, format="pdf",
        facecolor=(1.0, 1.0, 1.0, 0.0), transparent=False
    )
    rcParams["savefig.bbox"] = "tight"


#----------------------------------- Generic -----------------------------------

def sankey(
        left: List[str], right: List[str], title: str = "Sankey",
        left_color: Union[str, Callable[[str], str]] = "#E64B35",
        right_color: Union[str, Callable[[str], str]] = "#4EBBD5",
        link_color: Union[str, Callable[[pd.Series], str]] = "#CCCCCC",
        font_family: str = "Arial", font_size: float = 15.0,
        width: int = 400, height: int = 400,
        show: bool = True, embed_js: bool = False
) -> dict:
    r"""
    Make a sankey diagram

    Parameters
    ----------
    left
        Mapping source
    right
        Mapping target
    title
        Diagram title
    left_color
        Color of left nodes, either a single color or a mapping function
        that returns a color given the node name.
    right_color
        Color of right nodes, either a single color or a mapping function
        that returns a color given the node name.
    link_color
        Color of links, either a single color or a mapping function
        that returns a color given the link info.
    font_family
        Font family used for the plot
    font_size
        Font size for the plot
    width
        Graph width
    height
        Graph height
    show
        Whether to show interactive figure or only return the figure dict
    embed_js
        Whether to embed plotly.js library (only relevant when ``show=True``)

    Returns
    -------
    fig
        Figure dict that can be fed to :func:`plotly.offline.iplot`
        to show an interactive figure, or to :func:`plotly.io.write_image`
        to produce a static image file.

    Note
    ----
        If a mapping function is specified for ``link_color``,
        it should expect a :class:`pd.Series` object as the only argument,
        which contains the following fields:

        - left: the left node
        - right: the right node
        - value: population size connecting the two nodes
    """
    crosstab = pd.crosstab(
        pd.Series(left, name="left").astype(str),
        pd.Series(right, name="right").astype(str)
    ).reset_index().melt(id_vars=["left"]).sort_values("value")
    left_idx = pd.Index(np.unique(left))
    right_idx = pd.Index(np.unique(right))
    left_color = left_idx.map(left_color) if callable(left_color) \
        else [left_color] * left_idx.size
    right_color = right_idx.map(right_color) if callable(right_color) \
        else [right_color] * right_idx.size
    link_color = crosstab.apply(link_color, axis=1) if callable(link_color) \
        else [link_color] * crosstab.shape[0]

    sankey_data = dict(
        type="sankey",
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=np.concatenate([left_idx, right_idx]),
            color=np.concatenate([left_color, right_color])
        ),
        link=dict(
            source=left_idx.get_indexer(crosstab["left"]),
            target=right_idx.get_indexer(crosstab["right"]) + left_idx.size,
            value=crosstab["value"],
            color=link_color
        )
    )
    sankey_layout = dict(
        width=width,
        height=height,
        plot_bgcolor="rgba(1,1,1,0)",
        paper_bgcolor="rgba(1,1,1,0)",
        margin=dict(l=15, r=15, b=15, t=60),
        font=dict(family=font_family, size=font_size, color="#000000"),
        title=dict(text=title, x=0.5, xanchor="center", font_size=font_size + 2),
    )

    fig = dict(data=[sankey_data], layout=sankey_layout)
    if show:
        check_deps("plotly")
        import plotly.offline
        plotly.offline.init_notebook_mode(connected=not embed_js)
        plotly.offline.iplot(fig)
    return fig


def roc(
        true: np.ndarray, pred: np.ndarray, max_points: int = 500,
        ax: Optional[ma.Axes] = None, **kwargs
) -> ma.Axes:
    r"""
    Plot an ROC curve

    Parameters
    ----------
    true
        True labels
    pred
        Prediction values
    max_points
        Maximal number of points on the ROC curve, beyond which the points
        are equidistantly subsampled.
    ax
        Existing axes to plot on
    **kwargs
        Additional keyword arguments passed to :func:`seaborn.lineplot`

    Returns
    -------
    ax
        Plot axes
    """
    fpr, tpr, _ = sklearn.metrics.roc_curve(true, pred)
    idx = np.linspace(
        0, fpr.size, min(fpr.size, max_points), endpoint=False
    ).round().astype(int)
    idx[-1] = fpr.size - 1  # Always keep the last point
    data = pd.DataFrame({"FPR": fpr[idx], "TPR": tpr[idx]})
    ax = sns.lineplot(x="FPR", y="TPR", data=data, ax=ax, **kwargs)
    return ax


def prc(
        true: np.ndarray, pred: np.ndarray, max_points: int = 500,
        ax: Optional[ma.Axes] = None, **kwargs
) -> ma.Axes:
    r"""
    Plot a precision-recall curve

    Parameters
    ----------
    true
        True labels
    pred
        Prediction values
    max_points
        Maximal number of points on the precision-recall curve, beyond which
        the points are equidistantly subsampled.
    ax
        Existing axes to plot on
    **kwargs
        Additional keyword arguments passed to :func:`seaborn.lineplot`

    Returns
    -------
    ax
        Plot axes
    """
    prec, rec, _ = sklearn.metrics.precision_recall_curve(true, pred)
    idx = np.linspace(
        0, prec.size, min(prec.size, max_points), endpoint=False
    ).round().astype(int)
    idx[-1] = prec.size - 1  # Always keep the last point
    data = pd.DataFrame({"Precision": prec[idx], "Recall": rec[idx]})
    ax = sns.lineplot(x="Recall", y="Precision", data=data, ax=ax, **kwargs)
    return ax

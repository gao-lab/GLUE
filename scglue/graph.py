r"""
Graph-related functions
"""

from itertools import chain
from typing import Any, Callable, Iterable, Mapping, Optional, Set

import networkx as nx
from anndata import AnnData
from tqdm.auto import tqdm

from .utils import logged


def compose_multigraph(*graphs: nx.Graph) -> nx.MultiGraph:
    r"""
    Compose multi-graph from multiple graphs with no edge collision

    Parameters
    ----------
    graphs
        An arbitrary number of graphs to be composed from

    Returns
    -------
    composed
        Composed multi-graph

    Note
    ----
    The resulting multi-graph would be directed if any of the input graphs
    is directed.
    """
    if any(nx.is_directed(graph) for graph in graphs):
        graphs = [graph.to_directed() for graph in graphs]
        composed = nx.MultiDiGraph()
    else:
        composed = nx.MultiGraph()
    composed.add_edges_from(
        (e[0], e[1], graph.edges[e])
        for graph in graphs for e in graph.edges
    )
    return composed


def collapse_multigraph(
        graph: nx.MultiGraph, merge_fns: Optional[Mapping[str, Callable]] = None
) -> nx.Graph:
    r"""
    Collapse multi-edges into simple-edges

    Parameters
    ----------
    graph
        Input multi-graph
    merge_fns
        Attribute-specific merge functions, indexed by attribute name.
        Each merge function should accept a list of values and return
        a single value.

    Returns
    -------
    collapsed
        Collapsed graph

    Note
    ----
    The collapsed graph would be directed if the input graph is directed.
    Edges causing ValueError in ``merge_fns`` will be discarded.
    """
    if nx.is_directed(graph):  # MultiDiGraph
        collapsed = nx.DiGraph(graph)
    else:  # MultiGraph
        collapsed = nx.Graph(graph)
    if not merge_fns:
        return collapsed
    for e in tqdm(list(collapsed.edges), desc="collapse_multigraph"):
        attrs = graph.get_edge_data(*e).values()
        for k, fn in merge_fns.items():
            try:
                collapsed.edges[e][k] = fn([attr[k] for attr in attrs])
            except ValueError:
                collapsed.remove_edge(*e)
    return collapsed


def reachable_vertices(graph: nx.Graph, source: Iterable[Any]) -> Set[Any]:
    r"""
    Identify vertices reachable from source vertices
    (including source vertices themselves)

    Parameters
    ----------
    graph
        Input graph
    source
        Source vertices

    Returns
    -------
    reachable_vertices
        Reachable vertices
    """
    source = set(source)
    return set(chain.from_iterable(
        nx.descendants(graph, item) for item in source
        if graph.has_node(item)
    )).union(source)


@logged
def check_graph(
        graph: nx.Graph, adatas: Iterable[AnnData],
        cov: str = "error", attr: str = "error",
        loop: str = "error", sym: str = "error"
) -> None:
    r"""
    Check if a graph is a valid guidance graph

    Parameters
    ----------
    graph
        Graph to be checked
    adatas
        AnnData objects where graph nodes are variables
    cov
        Action to take if graph nodes does not cover all variables,
        must be one of {"ignore", "warn", "error"}
    attr
        Action to take if graph edges does not contain required attributes,
        must be one of {"ignore", "warn", "error"}
    loop
        Action to take if graph does not contain self-loops,
        must be one of {"ignore", "warn", "error"}
    sym
        Action to take if graph is not symmetric,
        must be one of {"ignore", "warn", "error"}
    """
    passed = True

    check_graph.logger.info("Checking variable coverage...")
    if not all(
        all(graph.has_node(var_name) for var_name in adata.var_names)
        for adata in adatas
    ):
        passed = False
        msg = "Some variables are not covered by the graph!"
        if cov == "error":
            raise ValueError(msg)
        elif cov == "warn":
            check_graph.logger.warning(msg)
        elif cov != "ignore":
            raise ValueError(f"Invalid `cov`: {cov}")

    check_graph.logger.info("Checking edge attributes...")
    if not all(
        "weight" in edge_attr and "sign" in edge_attr
        for edge_attr in dict(graph.edges).values()
    ):
        passed = False
        msg = "Missing weight or sign as edge attribute!"
        if attr == "error":
            raise ValueError(msg)
        elif attr == "warn":
            check_graph.logger.warning(msg)
        elif cov != "ignore":
            raise ValueError(f"Invalid `attr`: {attr}")

    check_graph.logger.info("Checking self-loops...")
    if not all(
        graph.has_edge(node, node) for node in graph.nodes
    ):
        passed = False
        msg = "Missing self-loop!"
        if loop == "error":
            raise ValueError(msg)
        elif loop == "warn":
            check_graph.logger.warning(msg)
        elif loop != "ignore":
            raise ValueError(f"Invalid `loop`: {loop}")

    check_graph.logger.info("Checking graph symmetry...")
    if not all(
        graph.has_edge(e[1], e[0]) for e in graph.edges
    ):
        passed = False
        msg = "Graph is not symmetric!"
        if sym == "error":
            raise ValueError(msg)
        elif sym == "warn":
            check_graph.logger.warning(msg)
        elif sym != "ignore":
            raise ValueError(f"Invalid `sym`: {sym}")

    if passed:
        check_graph.logger.info("All checks passed!")

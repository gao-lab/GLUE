r"""
Graph-related functions
"""

from itertools import chain
from typing import Any, Callable, Iterable, Mapping, Optional, Set

import networkx as nx

from .utils import smart_tqdm


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
    for e in smart_tqdm(list(collapsed.edges)):
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

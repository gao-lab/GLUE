r"""
Tests for the :mod:`scglue.graph` module
"""

# pylint: disable=redefined-outer-name, wildcard-import, unused-wildcard-import

import networkx as nx

import scglue

from .fixtures import *
from .utils import cmp_graphs


def test_compose_multigraph(graph, composed_graph):
    result = scglue.graph.compose_multigraph(graph, nx.Graph([
        ("f", "g", dict(dist=5, weight=1 / 5)),
        ("a", "b", dict(dist=2, weight=1 / 2))
    ]))
    cmp_graphs(result, composed_graph)
    result = scglue.graph.compose_multigraph(nx.Graph(graph), nx.Graph([
        ("f", "g", dict(dist=5, weight=1 / 5)),
        ("a", "b", dict(dist=2, weight=1 / 2))
    ]))
    cmp_graphs(result, nx.MultiGraph(composed_graph))

def test_collapse_multigraph(composed_graph):
    result = scglue.graph.collapse_multigraph(composed_graph)
    result = scglue.graph.collapse_multigraph(
        composed_graph, merge_fns={"dist": min, "weight": scglue.num.prob_or}
    )
    collapsed_graph = nx.DiGraph(composed_graph)
    collapsed_graph.edges["a", "b"]["dist"] = 2
    collapsed_graph.edges["b", "a"]["dist"] = 2
    collapsed_graph.edges["a", "b"]["weight"] = 1 - (10 / 11) * (1 / 2)
    collapsed_graph.edges["b", "a"]["weight"] = 1 - (10 / 11) * (1 / 2)
    for e in result.edges:
        result.edges[e]["weight"] = round(result.edges[e]["weight"], 5)
    for e in collapsed_graph.edges:
        collapsed_graph.edges[e]["weight"] = round(collapsed_graph.edges[e]["weight"], 5)
    cmp_graphs(result, collapsed_graph)

    def fn(x):
        raise ValueError
    result = scglue.graph.collapse_multigraph(
        nx.MultiGraph(composed_graph), merge_fns={"dist": min, "weight": fn}
    )
    assert len(result.edges) == 0

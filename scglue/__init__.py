r"""
GLUE (Graph-Linked Unified Embedding)
"""

try:
    from importlib.metadata import version
except ModuleNotFoundError:
    from pkg_resources import get_distribution
    version = lambda name: get_distribution(name).version

from . import data, genomics, graph, models, num, plot
from .utils import config, log


name = "scglue"
__version__ = version(name)

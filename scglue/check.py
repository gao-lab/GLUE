r"""
Dependency checking
"""

import importlib
import re
import types
from abc import abstractmethod
from typing import Optional

from packaging.version import parse

from . import version
from .utils import config, run_command


class Checker:

    r"""
    Checks availability and version of a dependency

    Parameters
    ----------
    name
        Name of the dependency
    vmin
        Minimal required version
    install_hint
        Install hint message to be printed if dependency is unavailable
    """

    def __init__(
            self, name: str, vmin: Optional[str] = None,
            install_hint: Optional[str] = None
    ) -> None:
        self.name = name
        self.vmin = parse(vmin) if vmin else vmin
        vreq = f" (>={self.vmin})" if self.vmin else ""
        self.vreq_hint = f"This function relies on {self.name}{vreq}."
        self.install_hint = install_hint

    @abstractmethod
    def check(self) -> None:
        r"""
        Check availability and version
        """
        raise NotImplementedError  # pragma: no cover


class ModuleChecker(Checker):

    r"""
    Checks availability and version of a Python module dependency

    Parameters
    ----------
    name
        Name of the dependency
    vmin
        Minimal required version
    install_hint
        Install hint message to be printed if dependency is unavailable
    """

    def __init__(
            self, name: str, package_name: Optional[str] = None,
            vmin: Optional[str] = None, install_hint: Optional[str] = None
    ) -> None:
        super().__init__(name, vmin, install_hint)
        self.package_name = package_name or name

    def check(self) -> None:
        try:
            importlib.import_module(self.name)
        except ModuleNotFoundError as e:
            raise RuntimeError(" ".join([self.vreq_hint, self.install_hint])) from e
        v = parse(version(self.package_name))
        if self.vmin and v < self.vmin:
            raise RuntimeError(" ".join([
                self.vreq_hint, f"Detected version is {v}.",
                "Please install a newer version.", self.install_hint
            ]))


class CmdChecker(Checker):

    r"""
    Checks availability and version of a command line dependency

    Parameters
    ----------
    name
        Name of the dependency
    cmd
        Command used to check version
    vregex
        Regular expression used to extract version from command output
    vmin
        Minimal required version
    install_hint
        Install hint message to be printed if dependency is unavailable
    """

    def __init__(
            self, name: str, cmd: str, vregex: str,
            vmin: Optional[str] = None, install_hint: Optional[str] = None
    ) -> None:
        super().__init__(name, vmin=vmin, install_hint=install_hint)
        self.cmd = cmd
        self.vregex = vregex

    def check(self) -> None:
        output_lines = run_command(
            self.cmd, log_command=False, print_output=False,
            err_message={
                127: " ".join([self.vreq_hint, self.install_hint])
            }
        )
        for output_line in output_lines:
            v = re.search(self.vregex, output_line)
            if v:
                v = parse(v.groups()[0])
                break
        else:
            v = None
        if self.vmin and v < self.vmin:
            raise RuntimeError(" ".join([
                self.vreq_hint, f"Detected version is {v}.",
                "Please install a newer version.", self.install_hint
            ]))


INSTALL_HINTS = types.SimpleNamespace(
    bedtools=
        "You may install bedtools following the guide from "
        "https://bedtools.readthedocs.io/en/latest/content/installation.html, "
        "or use `conda install -c bioconda bedtools` "
        "if a conda environment is being used.",
    plotly=
        "You may install plotly following the guide from "
        "https://plotly.com/python/getting-started/, "
        "or use `conda install -c plotly plotly` "
        "if a conda environment is being used."
)


CHECKERS = dict(
    bedtools=CmdChecker(
        "bedtools", f"{config.BEDTOOLS_PATH or 'bedtools'} --version", r"v([0-9\.]+)",
        vmin="2.29.2", install_hint=INSTALL_HINTS.bedtools
    ),
    plotly=ModuleChecker(
        "plotly",
        vmin=None, install_hint=INSTALL_HINTS.plotly
    )
)


def check_deps(*args) -> None:
    r"""
    Check whether certain dependencies are installed

    Parameters
    ----------
    args
        A list of dependencies to check
    """
    for item in args:
        CHECKERS[item].check()

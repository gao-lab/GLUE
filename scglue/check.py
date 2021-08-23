r"""
Dependency checking
"""

import re
import types
from typing import Optional

import packaging.version

from .utils import run_command


class Checker:

    r"""
    Checks availability and version of a dependency

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
        self.name = name
        self.cmd = cmd
        self.vregex = vregex
        self.vmin = packaging.version.parse(vmin) if vmin else vmin

        vreq = f" (>= {self.vmin})" if self.vmin else ""
        self.vreq_hint = f"This function relies on {self.name}{vreq}."
        self.install_hint = install_hint

    def check(self) -> None:
        r"""
        Check availability and version
        """
        output_lines = run_command(
            self.cmd, log_command=False, print_output=False,
            err_message={
                127: " ".join([self.vreq_hint, self.install_hint])
            }
        )
        version = packaging.version.parse(re.search(
            self.vregex, "\n".join(output_lines)
        ).groups()[0])
        if self.vmin and version < self.vmin:
            raise RuntimeError(" ".join([
                self.vreq_hint, f"Detected version is {version}.",
                "Please install a newer version.", self.install_hint
            ]))


INSTALL_HINTS = types.SimpleNamespace(
    meme=
        "You may install meme suite following the guide from "
        "http://meme-suite.org/doc/install.html, "
        "or use `conda install -c bioconda meme` "
        "if a conda environment is being used.",
    bedtools=
        "You may install bedtools following the guide from "
        "https://bedtools.readthedocs.io/en/latest/content/installation.html, "
        "or use `conda install -c bioconda bedtools` "
        "if a conda environment is being used."
)


CHECKERS = dict(
    bedtools=Checker(
        "bedtools", "bedtools --version", r"v([0-9\.]+)",
        vmin="2.29.2", install_hint=INSTALL_HINTS.bedtools
    ),
    fimo=Checker(
        "fimo", "fimo --version", r"([0-9\.]+)",
        vmin=None, install_hint=INSTALL_HINTS.meme
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

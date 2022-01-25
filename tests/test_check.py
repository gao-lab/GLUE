r"""
Tests for the :mod:`scglue.check` module
"""

import pytest

import scglue.check


def test_module_checker():
    module_checker = scglue.check.ModuleChecker(
        "numpy",
        vmin=None, install_hint="You may install via..."
    )
    module_checker.check()

    module_checker = scglue.check.ModuleChecker(
        "numpy",
        vmin="99.99.99", install_hint="You may install via..."
    )
    with pytest.raises(RuntimeError):
        module_checker.check()

    module_checker = scglue.check.ModuleChecker(
        "xxx",
        vmin=None, install_hint="You may install via..."
    )
    with pytest.raises(RuntimeError):
        module_checker.check()


def test_cmd_checker():
    cmd_checker = scglue.check.CmdChecker(
        "ls", "ls --version", r"([0-9\.]+)$",
        vmin=None, install_hint="You may install via..."
    )
    cmd_checker.check()

    cmd_checker = scglue.check.CmdChecker(
        "ls", "ls --version", r"([0-9\.]+)$",
        vmin="99.99.99", install_hint="You may install via..."
    )
    with pytest.raises(RuntimeError):
        cmd_checker.check()

    cmd_checker = scglue.check.CmdChecker(
        "xxx", "xxx --version", r"([0-9\.]+)$",
        vmin=None, install_hint="You may install via..."
    )
    with pytest.raises(RuntimeError):
        cmd_checker.check()

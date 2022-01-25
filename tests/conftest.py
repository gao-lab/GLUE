r"""
Test configuration
"""

# pylint: disable=missing-function-docstring

import logging

import numpy as np
import pytest
import torch

import scglue


def pytest_addoption(parser):
    parser.addoption("--loader-workers", dest="loader_workers", type=int, default=0)  # Slow if not 0, due to wait for pytorch subprocesses to join
    parser.addoption("--cpu-only", dest="cpu_only", action="store_true", default=False)


def pytest_generate_tests(metafunc):
    scglue.log.console_log_level = logging.DEBUG
    scglue.log.file_log_level = logging.DEBUG
    scglue.config.PRINT_LOSS_INTERVAL = 2
    scglue.config.DATALOADER_NUM_WORKERS = metafunc.config.option.loader_workers
    scglue.config.CPU_ONLY = metafunc.config.option.cpu_only
    np.random.seed(123)
    torch.manual_seed(321)


def pytest_configure(config):
    config.addinivalue_line("markers", "cpu_only: mark test to run only in cpu-only mode")


def pytest_collection_modifyitems(config, items):
    if not config.option.cpu_only:
        skip_cpu_only = pytest.mark.skip(reason="only runs in cpu-only mode")
        for item in items:
            if "cpu_only" in item.keywords:
                item.add_marker(skip_cpu_only)

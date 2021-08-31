r"""
Utility functions for snakemake files
"""

# pylint: disable=missing-function-docstring, redefined-outer-name

from functools import reduce
from operator import add
from pathlib import Path


def conf_expand_pattern(conf, placeholder="null"):
    expand_pattern = "-".join(f"{key}:{{{key}}}" for key in conf)
    return expand_pattern if expand_pattern else placeholder


def expand(pattern, **wildcards):
    from snakemake.io import expand

    has_default_choices = False
    for val in wildcards.values():  # Sanity check
        if isinstance(val, dict):
            if "default" not in val or "choices" not in val:
                print(val)
                raise ValueError("Invalid default choices!")
            has_default_choices = True

    if not has_default_choices:
        return expand(pattern, **wildcards)

    expand_set = set()
    for key, val in wildcards.items():
        if isinstance(val, dict):
            wildcards_use = {key: val["choices"]}
            for other_key, other_val in wildcards.items():
                if other_key == key:
                    continue
                if isinstance(other_val, dict):
                    wildcards_use[other_key] = other_val["default"]
                else:
                    wildcards_use[other_key] = other_val
            expand_set = expand_set.union(expand(pattern, **wildcards_use))
    return list(expand_set)


def seed2range(config):
    for key, val in config.items():
        if isinstance(val, dict):
            seed2range(val)
        elif key.endswith("seed") and val != 0:
            config[key] = range(val)


def target_directories(config):
    seed2range(config)

    dataset = config["dataset"].keys()
    subsample_conf = config["subsample"] or {}
    subsample_conf = expand(
        conf_expand_pattern(subsample_conf, placeholder="original"),
        **subsample_conf
    )

    def per_method(method):
        prior_conf = config["prior"] or {}
        prior_conf = {} if method in ("UnionCom", "iNMF_FiG", "LIGER_FiG") else prior_conf  # Methods that do not use prior feature matching
        prior_conf = expand(
            conf_expand_pattern(prior_conf, placeholder="null"),
            **prior_conf
        )
        hyperparam_conf = config["method"][method] or {}
        hyperparam_conf = expand(
            conf_expand_pattern(hyperparam_conf, placeholder="default"),
            **hyperparam_conf
        )
        seed = 0 if method in ("bindSC", ) else config["seed"]  # Methods that are deterministic
        return expand(
            "results/raw/{dataset}/{subsample_conf}/{prior_conf}/{method}/{hyperparam_conf}/seed:{seed}",
            dataset=dataset,
            subsample_conf=subsample_conf,
            prior_conf=prior_conf,
            method=method,
            hyperparam_conf=hyperparam_conf,
            seed=seed
        )

    return reduce(add, map(per_method, config["method"]))


def target_files(directories):

    def per_directory(directory):
        directory = Path(directory)
        if (directory / ".blacklist").exists():
            return []
        return [
            directory / "metrics.yaml",
            directory / "cell_type.pdf",
            directory / "domain.pdf"
        ]

    return reduce(add, map(per_directory, directories))

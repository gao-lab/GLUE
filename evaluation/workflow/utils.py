r"""
Utility functions for snakemake files
"""

# pylint: disable=missing-function-docstring, redefined-outer-name

from functools import reduce
from operator import add
from pathlib import Path

import parse


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
    data_conf = config["data"] or {}
    data_conf = expand(
        conf_expand_pattern(data_conf, placeholder="original"),
        **data_conf
    )

    def per_method(method):
        prior_conf = config["prior"] or {}
        prior_conf = {} if method in (
            "UnionCom", "Pamona", "MMD_MA",
            "iNMF_FiG", "LIGER_FiG"  # Methods that do not use prior feature matching
        ) else prior_conf
        prior_conf = expand(
            conf_expand_pattern(prior_conf, placeholder="null"),
            **prior_conf
        )
        hyperparam_conf = config["method"][method] or {}
        hyperparam_conf = expand(
            conf_expand_pattern(hyperparam_conf, placeholder="default"),
            **hyperparam_conf
        )
        seed = 0 if method in ("Pamona", "bindSC", "Harmony") else config["seed"]  # Methods that are deterministic
        return expand(
            "results/raw/{dataset}/{data_conf}/{prior_conf}/{method}/{hyperparam_conf}/seed:{seed}",
            dataset=dataset,
            data_conf=data_conf,
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


def default_prior_conf(prior_conf):
    pattern = "gene_region:{gene_region}-extend_range:{extend_range}-corrupt_rate:{corrupt_rate}-corrupt_seed:{corrupt_seed}"
    conf = parse.parse(pattern, prior_conf).named
    conf.update({"corrupt_rate": 0.0, "corrupt_seed": 0})
    return pattern.format(**conf)


def default_hyperparam_conf(hyperparam_conf):
    pattern = "dim:{dim}-alt_dim:{alt_dim}-hidden_depth:{hidden_depth}-hidden_dim:{hidden_dim}-dropout:{dropout}-lam_graph:{lam_graph}-lam_align:{lam_align}-neg_samples:{neg_samples}"
    conf = parse.parse(pattern, hyperparam_conf).named
    conf.update({
        "dim": 50,
        "alt_dim": 100,
        "hidden_depth": 2,
        "hidden_dim": 256,
        "dropout": 0.2,
        "lam_graph": 0.02,
        "lam_align": 0.05,
        "neg_samples": 10
    })
    return pattern.format(**conf)

from copy import deepcopy
from utils import target_directories, target_files

config_baseline = deepcopy(config)
config_baseline["prior"]["corrupt_rate"] = 0.0
config_baseline["prior"]["corrupt_seed"] = 0

rule plot:
    input:
        "results/prior.csv"
    output:
        "results/prior.pdf"
    threads: 1
    script:
        "scripts/prior.R"

rule summarize:
    input:
        target_files(target_directories(config) + target_directories(config_baseline))
    output:
        "results/prior.csv"
    params:
        pattern=lambda wildcards: "results/raw/{dataset}/{subsample_conf}/gene_region:{gene_region}-extend_range:{extend_range:d}-corrupt_rate:{corrupt_rate:f}-corrupt_seed:{corrupt_seed:d}/{method}/{hyperparam_conf}/seed:{seed:d}/metrics.yaml"
    threads: 1
    script:
        "scripts/summarize.py"

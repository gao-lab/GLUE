from copy import deepcopy
from utils import target_directories, target_files

config_baseline = deepcopy(config)
config_baseline["prior"]["corrupt_rate"] = 0.0
config_baseline["prior"]["corrupt_seed"] = 0

rule plot:
    input:
        "results/benchmark.csv",
        "results/prior_corrupt.csv"
    output:
        foscttm="results/prior_corrupt_foscttm.pdf",
        map="results/prior_corrupt_map.pdf",
        sas="results/prior_corrupt_sas.pdf",
        asw="results/prior_corrupt_asw.pdf",
        aswb="results/prior_corrupt_aswb.pdf",
        nc="results/prior_corrupt_nc.pdf",
        gc="results/prior_corrupt_gc.pdf",
        bio="results/prior_corrupt_bio.pdf",
        int="results/prior_corrupt_int.pdf",
        overall="results/prior_corrupt_overall.pdf",
        fc="results/prior_corrupt_fc.pdf"
    threads: 1
    script:
        "scripts/prior_corrupt.R"

rule summarize:
    input:
        target_files(target_directories(config) + target_directories(config_baseline))
    output:
        "results/prior_corrupt.csv"
    params:
        pattern=lambda wildcards: "results/raw/{dataset}/{data_conf}/gene_region:{gene_region}-extend_range:{extend_range:d}-corrupt_rate:{corrupt_rate:f}-corrupt_seed:{corrupt_seed:d}/{method}/{hyperparam_conf}/seed:{seed:d}/metrics.yaml"
    threads: 1
    script:
        "scripts/summarize.py"

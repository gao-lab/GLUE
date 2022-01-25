from copy import deepcopy
from utils import target_directories, target_files


def select_combinations(x):
    if "gene_region:combined-extend_range:0" in x:
        return True
    if "gene_region:promoter-extend_range:150000" in x:
        return True
    return False


rule plot:
    input:
        "results/benchmark.csv",
        "results/prior_alt.csv"
    output:
        foscttm="results/prior_alt_foscttm.pdf",
        map="results/prior_alt_map.pdf",
        sas="results/prior_alt_sas.pdf",
        asw="results/prior_alt_asw.pdf",
        aswb="results/prior_alt_aswb.pdf",
        nc="results/prior_alt_nc.pdf",
        gc="results/prior_alt_gc.pdf",
        bio="results/prior_alt_bio.pdf",
        int="results/prior_alt_int.pdf",
        overall="results/prior_alt_overall.pdf"
    threads: 1
    script:
        "scripts/prior_alt.R"

rule summarize:
    input:
        target_files(filter(select_combinations, target_directories(config)))
    output:
        "results/prior_alt.csv"
    params:
        pattern=lambda wildcards: "results/raw/{dataset}/{data_conf}/gene_region:{gene_region}-extend_range:{extend_range:d}-corrupt_rate:{corrupt_rate:f}-corrupt_seed:{corrupt_seed:d}/{method}/{hyperparam_conf}/seed:{seed:d}/metrics.yaml"
    threads: 1
    script:
        "scripts/summarize.py"

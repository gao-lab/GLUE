from utils import target_directories, target_files

rule plot:
    input:
        "results/subsample.csv"
    output:
        "results/subsample.pdf"
    threads: 1
    script:
        "scripts/subsample.R"

rule summarize:
    input:
        target_files(target_directories(config))
    output:
        "results/subsample.csv"
    params:
        pattern=lambda wildcards: "results/raw/{dataset}/subsample_size:{subsample_size:d}-subsample_seed:{subsample_seed:d}/{prior_conf}/{method}/{hyperparam_conf}/seed:{seed:d}/metrics.yaml"
    threads: 1
    script:
        "scripts/summarize.py"

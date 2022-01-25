from utils import target_directories, target_files

rule plot:
    input:
        "results/benchmark.csv",
        "results/hvg.csv"
    output:
        foscttm="results/hvg_foscttm.pdf",
        map="results/hvg_map.pdf",
        sas="results/hvg_sas.pdf",
        asw="results/hvg_asw.pdf",
        aswb="results/hvg_aswb.pdf",
        nc="results/hvg_nc.pdf",
        gc="results/hvg_gc.pdf",
        bio="results/hvg_bio.pdf",
        int="results/hvg_int.pdf",
        overall="results/hvg_overall.pdf"
    threads: 1
    script:
        "scripts/hvg.R"

rule summarize:
    input:
        target_files(target_directories(config))
    output:
        "results/hvg.csv"
    params:
        pattern=lambda wildcards: "results/raw/{dataset}/hvg:{hvg:d}/{prior_conf}/{method}/{hyperparam_conf}/seed:{seed:d}/metrics.yaml"
    threads: 1
    script:
        "scripts/summarize.py"

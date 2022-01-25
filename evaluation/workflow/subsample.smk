from utils import target_directories, target_files

rule plot:
    input:
        "results/benchmark.csv",
        "results/subsample.csv"
    output:
        foscttm="results/subsample_foscttm.pdf",
        map="results/subsample_map.pdf",
        sas="results/subsample_sas.pdf",
        asw="results/subsample_asw.pdf",
        aswb="results/subsample_aswb.pdf",
        nc="results/subsample_nc.pdf",
        gc="results/subsample_gc.pdf",
        bio="results/subsample_bio.pdf",
        int="results/subsample_int.pdf",
        overall="results/subsample_overall.pdf",
        fc="results/subsample_fc.pdf"
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

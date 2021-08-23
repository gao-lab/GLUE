from utils import target_directories, target_files

rule plot:
    input:
        "results/benchmark.csv"
    output:
        foscttm="results/benchmark_foscttm.pdf",
        map_vs_sas="results/benchmark_map_vs_sas.pdf"
    threads: 1
    script:
        "scripts/benchmark.R"

rule umap_grid:
    input:
        "results/benchmark.csv"
    output:
        directory("results/umap_grid")
    params:
        rna_umap=lambda wildcards: "results/raw/{dataset}/{subsample_conf}/{prior_conf}/{method}/{hyperparam_conf}/seed:{seed}/rna_umap.csv",
        atac_umap=lambda wildcards: "results/raw/{dataset}/{subsample_conf}/{prior_conf}/{method}/{hyperparam_conf}/seed:{seed}/atac_umap.csv"
    threads: 1
    script:
        "scripts/umap_grid.py"

rule summarize:
    input:
        target_files(target_directories(config))
    output:
        "results/benchmark.csv"
    params:
        pattern=lambda wildcards: "results/raw/{dataset}/{subsample_conf}/{prior_conf}/{method}/{hyperparam_conf}/seed:{seed:d}/metrics.yaml"
    threads: 1
    script:
        "scripts/summarize.py"

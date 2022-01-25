from utils import target_directories, target_files

rule plot:
    input:
        "results/benchmark.csv"
    output:
        foscttm="results/benchmark_foscttm.pdf",
        map_vs_sas="results/benchmark_map_vs_sas.pdf",
        map_vs_sas_hull="results/benchmark_map_vs_sas_hull.pdf",
        asw_vs_aswb="results/benchmark_asw_vs_aswb.pdf",
        asw_vs_aswb_hull="results/benchmark_asw_vs_aswb_hull.pdf",
        nc_vs_gc="results/benchmark_nc_vs_gc.pdf",
        nc_vs_gc_hull="results/benchmark_nc_vs_gc_hull.pdf",
        bio_vs_int="results/benchmark_bio_vs_int.pdf",
        bio_vs_int_hull="results/benchmark_bio_vs_int_hull.pdf",
        overall="results/benchmark_overall.pdf",
        legend="results/benchmark_legend.pdf"
    threads: 1
    script:
        "scripts/benchmark.R"

rule umap_grid:
    input:
        "results/benchmark.csv"
    output:
        directory("results/umap_grid")
    params:
        rna_umap=lambda wildcards: "results/raw/{dataset}/{data_conf}/{prior_conf}/{method}/{hyperparam_conf}/seed:{seed}/rna_umap.csv",
        atac_umap=lambda wildcards: "results/raw/{dataset}/{data_conf}/{prior_conf}/{method}/{hyperparam_conf}/seed:{seed}/atac_umap.csv"
    threads: 1
    script:
        "scripts/umap_grid.py"

rule summarize:
    input:
        target_files(target_directories(config))
    output:
        "results/benchmark.csv"
    params:
        pattern=lambda wildcards: "results/raw/{dataset}/{data_conf}/{prior_conf}/{method}/{hyperparam_conf}/seed:{seed:d}/metrics.yaml"
    threads: 1
    script:
        "scripts/summarize.py"

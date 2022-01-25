from utils import target_directories, target_files

rule plot:
    input:
        "results/benchmark_noprep.csv"
    output:
        foscttm="results/benchmark_noprep_foscttm.pdf",
        map_vs_sas="results/benchmark_noprep_map_vs_sas.pdf",
        map_vs_sas_hull="results/benchmark_noprep_map_vs_sas_hull.pdf",
        asw_vs_aswb="results/benchmark_noprep_asw_vs_aswb.pdf",
        asw_vs_aswb_hull="results/benchmark_noprep_asw_vs_aswb_hull.pdf",
        nc_vs_gc="results/benchmark_noprep_nc_vs_gc.pdf",
        nc_vs_gc_hull="results/benchmark_noprep_nc_vs_gc_hull.pdf",
        bio_vs_int="results/benchmark_noprep_bio_vs_int.pdf",
        bio_vs_int_hull="results/benchmark_noprep_bio_vs_int_hull.pdf",
        overall="results/benchmark_noprep_overall.pdf",
        legend="results/benchmark_noprep_legend.pdf"
    threads: 1
    script:
        "scripts/benchmark.R"

rule summarize:
    input:
        target_files(target_directories(config))
    output:
        "results/benchmark_noprep.csv"
    params:
        pattern=lambda wildcards: "results/raw/{dataset}/{data_conf}/{prior_conf}/{method}/{hyperparam_conf}/seed:{seed:d}/metrics.yaml"
    threads: 1
    script:
        "scripts/summarize.py"

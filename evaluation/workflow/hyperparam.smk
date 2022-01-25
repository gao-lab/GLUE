from utils import target_directories, target_files

rule plot:
    input:
        "results/benchmark.csv",
        "results/hyperparam.csv"
    output:
        foscttm="results/hyperparam_foscttm.pdf",
        map="results/hyperparam_map.pdf",
        sas="results/hyperparam_sas.pdf",
        asw="results/hyperparam_asw.pdf",
        aswb="results/hyperparam_aswb.pdf",
        nc="results/hyperparam_nc.pdf",
        gc="results/hyperparam_gc.pdf",
        bio="results/hyperparam_bio.pdf",
        int="results/hyperparam_int.pdf",
        overall="results/hyperparam_overall.pdf",
        fc="results/hyperparam_fc.pdf",
        legend_h="results/hyperparam_legend_h.pdf",
        legend_v="results/hyperparam_legend_v.pdf"
    threads: 1
    script:
        "scripts/hyperparam.R"

rule summarize:
    input:
        target_files(target_directories(config))
    output:
        "results/hyperparam.csv"
    params:
        pattern=lambda wildcards: "results/raw/{dataset}/{data_conf}/{prior_conf}/GLUE/dim:{dim:d}-alt_dim:{alt_dim:d}-hidden_depth:{hidden_depth:d}-hidden_dim:{hidden_dim:d}-dropout:{dropout:f}-lam_graph:{lam_graph:f}-lam_align:{lam_align:f}-neg_samples:{neg_samples:d}/seed:{seed:d}/metrics.yaml"
    threads: 1
    script:
        "scripts/summarize.py"

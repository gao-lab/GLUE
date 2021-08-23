from utils import target_directories, target_files

rule plot:
    input:
        "results/hyperparam.csv"
    output:
        "results/hyperparam.pdf"
    threads: 1
    script:
        "scripts/hyperparam.R"

rule summarize:
    input:
        target_files(target_directories(config))
    output:
        "results/hyperparam.csv"
    params:
        pattern=lambda wildcards: "results/raw/{dataset}/{subsample_conf}/{prior_conf}/GLUE/dim:{dim:d}-alt_dim:{alt_dim:d}-hidden_depth:{hidden_depth:d}-hidden_dim:{hidden_dim:d}-dropout:{dropout:f}-lam_graph:{lam_graph:f}-lam_align:{lam_align:f}-neg_samples:{neg_samples:d}/seed:{seed:d}/metrics.yaml"
    threads: 1
    script:
        "scripts/summarize.py"

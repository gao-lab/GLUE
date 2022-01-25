import os

rule run_UnionCom:
    input:
        rna="{path}/rna.h5ad",
        atac="{path}/atac.h5ad"
    output:
        rna_latent="{path}/null/UnionCom/default/seed:{seed}/rna_latent.csv",
        atac_latent="{path}/null/UnionCom/default/seed:{seed}/atac_latent.csv",
        run_info="{path}/null/UnionCom/default/seed:{seed}/run_info.yaml"
    log:
        "{path}/null/UnionCom/default/seed:{seed}/run_UnionCom.log"
    params:
        blacklist="{path}/null/UnionCom/default/seed:{seed}/.blacklist"
    conda: "../envs/UnionCom.yaml"
    threads: 4
    resources: gpu=1
    shell:
        "timeout {config[timeout]} python -u workflow/scripts/run_UnionCom.py "
        "--input-rna {input.rna} --input-atac {input.atac} -s {wildcards.seed} "
        "--output-rna {output.rna_latent} --output-atac {output.atac_latent} --run-info {output.run_info} "
        "> {log} 2>&1 || touch {params.blacklist}"

rule run_Pamona:
    input:
        rna="{path}/rna.h5ad",
        atac="{path}/atac.h5ad"
    output:
        rna_latent="{path}/null/Pamona/default/seed:{seed}/rna_latent.csv",
        atac_latent="{path}/null/Pamona/default/seed:{seed}/atac_latent.csv",
        run_info="{path}/null/Pamona/default/seed:{seed}/run_info.yaml"
    log:
        "{path}/null/Pamona/default/seed:{seed}/run_Pamona.log"
    params:
        blacklist="{path}/null/Pamona/default/seed:{seed}/.blacklist"
    threads: 4
    shell:
        "timeout {config[timeout]} python -u workflow/scripts/run_Pamona.py "
        "--input-rna {input.rna} --input-atac {input.atac} -s {wildcards.seed} "
        "--output-rna {output.rna_latent} --output-atac {output.atac_latent} --run-info {output.run_info} "
        "> {log} 2>&1 || touch {params.blacklist}"

rule run_MMD_MA:
    input:
        rna="{path}/rna.h5ad",
        atac="{path}/atac.h5ad"
    output:
        rna_latent="{path}/null/MMD_MA/default/seed:{seed}/rna_latent.csv",
        atac_latent="{path}/null/MMD_MA/default/seed:{seed}/atac_latent.csv",
        run_info="{path}/null/MMD_MA/default/seed:{seed}/run_info.yaml"
    log:
        "{path}/null/MMD_MA/default/seed:{seed}/run_MMD_MA.log"
    params:
        blacklist="{path}/null/MMD_MA/default/seed:{seed}/.blacklist"
    threads: 4
    resources: gpu=1
    shell:
        "timeout {config[timeout]} python -u workflow/scripts/run_MMD_MA.py "
        "--input-rna {input.rna} --input-atac {input.atac} -s {wildcards.seed} "
        "--output-rna {output.rna_latent} --output-atac {output.atac_latent} --run-info {output.run_info} "
        "> {log} 2>&1 || touch {params.blacklist}"

rule run_iNMF:
    input:
        rna="{path}/rna.h5ad",
        atac="{path}/{prior_conf}/atac2rna.h5ad"
    output:
        rna_latent="{path}/{prior_conf}/iNMF/default/seed:{seed}/rna_latent.csv",
        atac_latent="{path}/{prior_conf}/iNMF/default/seed:{seed}/atac_latent.csv",
        run_info="{path}/{prior_conf}/iNMF/default/seed:{seed}/run_info.yaml"
    log:
        "{path}/{prior_conf}/iNMF/default/seed:{seed}/run_iNMF.log"
    params:
        blacklist="{path}/{prior_conf}/iNMF/default/seed:{seed}/.blacklist"
    threads: 4
    shell:
        "timeout {config[timeout]} Rscript workflow/scripts/run_iNMF.R "
        "--input-rna {input.rna} --input-atac {input.atac} -s {wildcards.seed} "
        "--output-rna {output.rna_latent} --output-atac {output.atac_latent} --run-info {output.run_info} "
        "> {log} 2>&1 || touch {params.blacklist}"

rule run_iNMF_FiG:
    input:
        rna="{path}/rna.h5ad",
        atac="{path}/frags2rna.h5ad"
    output:
        rna_latent="{path}/null/iNMF_FiG/default/seed:{seed}/rna_latent.csv",
        atac_latent="{path}/null/iNMF_FiG/default/seed:{seed}/atac_latent.csv",
        run_info="{path}/null/iNMF_FiG/default/seed:{seed}/run_info.yaml"
    log:
        "{path}/null/iNMF_FiG/default/seed:{seed}/run_iNMF_FiG.log"
    params:
        blacklist="{path}/null/iNMF_FiG/default/seed:{seed}/.blacklist"
    threads: 4
    shell:
        "timeout {config[timeout]} Rscript workflow/scripts/run_iNMF.R "
        "--input-rna {input.rna} --input-atac {input.atac} -s {wildcards.seed} "
        "--output-rna {output.rna_latent} --output-atac {output.atac_latent} --run-info {output.run_info} "
        "> {log} 2>&1 || touch {params.blacklist}"

rule run_LIGER:
    input:
        rna="{path}/rna.h5ad",
        atac="{path}/{prior_conf}/atac2rna.h5ad"
    output:
        rna_latent="{path}/{prior_conf}/LIGER/default/seed:{seed}/rna_latent.csv",
        atac_latent="{path}/{prior_conf}/LIGER/default/seed:{seed}/atac_latent.csv",
        run_info="{path}/{prior_conf}/LIGER/default/seed:{seed}/run_info.yaml"
    log:
        "{path}/{prior_conf}/LIGER/default/seed:{seed}/run_LIGER.log"
    params:
        blacklist="{path}/{prior_conf}/LIGER/default/seed:{seed}/.blacklist"
    threads: 4
    shell:
        "timeout {config[timeout]} Rscript workflow/scripts/run_LIGER.R "
        "--input-rna {input.rna} --input-atac {input.atac} -s {wildcards.seed} "
        "--output-rna {output.rna_latent} --output-atac {output.atac_latent} --run-info {output.run_info} "
        "> {log} 2>&1 || touch {params.blacklist}"

rule run_LIGER_FiG:
    input:
        rna="{path}/rna.h5ad",
        atac="{path}/frags2rna.h5ad"
    output:
        rna_latent="{path}/null/LIGER_FiG/default/seed:{seed}/rna_latent.csv",
        atac_latent="{path}/null/LIGER_FiG/default/seed:{seed}/atac_latent.csv",
        run_info="{path}/null/LIGER_FiG/default/seed:{seed}/run_info.yaml"
    log:
        "{path}/null/LIGER_FiG/default/seed:{seed}/run_LIGER_FiG.log"
    params:
        blacklist="{path}/null/LIGER_FiG/default/seed:{seed}/.blacklist"
    threads: 4
    shell:
        "timeout {config[timeout]} Rscript workflow/scripts/run_LIGER.R "
        "--input-rna {input.rna} --input-atac {input.atac} -s {wildcards.seed} "
        "--output-rna {output.rna_latent} --output-atac {output.atac_latent} --run-info {output.run_info} "
        "> {log} 2>&1 || touch {params.blacklist}"

rule run_Harmony:
    input:
        rna="{path}/rna.h5ad",
        atac="{path}/{prior_conf}/atac2rna.h5ad"
    output:
        rna_latent="{path}/{prior_conf}/Harmony/default/seed:{seed}/rna_latent.csv",
        atac_latent="{path}/{prior_conf}/Harmony/default/seed:{seed}/atac_latent.csv",
        run_info="{path}/{prior_conf}/Harmony/default/seed:{seed}/run_info.yaml"
    log:
        "{path}/{prior_conf}/Harmony/default/seed:{seed}/run_Harmony.log"
    params:
        blacklist="{path}/{prior_conf}/Harmony/default/seed:{seed}/.blacklist"
    threads: 4
    shell:
        "timeout {config[timeout]} Rscript workflow/scripts/run_Harmony.R "
        "--input-rna {input.rna} --input-atac {input.atac} -s {wildcards.seed} "
        "--output-rna {output.rna_latent} --output-atac {output.atac_latent} --run-info {output.run_info} "
        "> {log} 2>&1 || touch {params.blacklist}"

rule run_bindSC:
    input:
        rna="{path}/rna.h5ad",
        atac="{path}/atac.h5ad",
        atac2rna="{path}/{prior_conf}/atac2rna.h5ad"
    output:
        rna_latent="{path}/{prior_conf}/bindSC/default/seed:{seed}/rna_latent.csv",
        atac_latent="{path}/{prior_conf}/bindSC/default/seed:{seed}/atac_latent.csv",
        run_info="{path}/{prior_conf}/bindSC/default/seed:{seed}/run_info.yaml"
    log:
        "{path}/{prior_conf}/bindSC/default/seed:{seed}/run_bindSC.log"
    params:
        blacklist="{path}/{prior_conf}/bindSC/default/seed:{seed}/.blacklist"
    threads: 4
    shell:
        "timeout {config[timeout]} Rscript workflow/scripts/run_bindSC.R "
        "--input-rna {input.rna} --input-atac {input.atac} "
        "--input-atac2rna {input.atac2rna} -s {wildcards.seed} "
        "--output-rna {output.rna_latent} --output-atac {output.atac_latent} --run-info {output.run_info} "
        "> {log} 2>&1 || touch {params.blacklist}"

rule run_CCA_anchor:
    input:
        rna="{path}/rna.h5ad",
        atac="{path}/atac.h5ad",
        atac2rna="{path}/{prior_conf}/atac2rna.h5ad"
    output:
        rna_latent="{path}/{prior_conf}/CCA_anchor/default/seed:{seed}/rna_latent.csv",
        atac_latent="{path}/{prior_conf}/CCA_anchor/default/seed:{seed}/atac_latent.csv",
        run_info="{path}/{prior_conf}/CCA_anchor/default/seed:{seed}/run_info.yaml"
    log:
        "{path}/{prior_conf}/CCA_anchor/default/seed:{seed}/run_CCA_anchor.log"
    params:
        blacklist="{path}/{prior_conf}/CCA_anchor/default/seed:{seed}/.blacklist"
    threads: 4
    shell:
        "timeout {config[timeout]} Rscript workflow/scripts/run_CCA_anchor.R "
        "--input-rna {input.rna} --input-atac {input.atac} "
        "--input-atac2rna {input.atac2rna} -s {wildcards.seed} "
        "--output-rna {output.rna_latent} --output-atac {output.atac_latent} --run-info {output.run_info} "
        "> {log} 2>&1 || touch {params.blacklist}"

rule run_GLUE:
    input:
        rna="{path}/rna.h5ad",
        atac="{path}/atac.h5ad",
        prior="{path}/{prior_conf}/sub.graphml.gz"
    output:
        rna_latent="{path}/{prior_conf}/GLUE/dim:{dim}-alt_dim:{alt_dim}-hidden_depth:{hidden_depth}-hidden_dim:{hidden_dim}-dropout:{dropout}-lam_graph:{lam_graph}-lam_align:{lam_align}-neg_samples:{neg_samples}/seed:{seed}/rna_latent.csv",
        atac_latent="{path}/{prior_conf}/GLUE/dim:{dim}-alt_dim:{alt_dim}-hidden_depth:{hidden_depth}-hidden_dim:{hidden_dim}-dropout:{dropout}-lam_graph:{lam_graph}-lam_align:{lam_align}-neg_samples:{neg_samples}/seed:{seed}/atac_latent.csv",
        feature_latent="{path}/{prior_conf}/GLUE/dim:{dim}-alt_dim:{alt_dim}-hidden_depth:{hidden_depth}-hidden_dim:{hidden_dim}-dropout:{dropout}-lam_graph:{lam_graph}-lam_align:{lam_align}-neg_samples:{neg_samples}/seed:{seed}/feature_latent.csv",
        run_info="{path}/{prior_conf}/GLUE/dim:{dim}-alt_dim:{alt_dim}-hidden_depth:{hidden_depth}-hidden_dim:{hidden_dim}-dropout:{dropout}-lam_graph:{lam_graph}-lam_align:{lam_align}-neg_samples:{neg_samples}/seed:{seed}/run_info.yaml"
    log:
        "{path}/{prior_conf}/GLUE/dim:{dim}-alt_dim:{alt_dim}-hidden_depth:{hidden_depth}-hidden_dim:{hidden_dim}-dropout:{dropout}-lam_graph:{lam_graph}-lam_align:{lam_align}-neg_samples:{neg_samples}/seed:{seed}/run_GLUE.log"
    params:
        train_dir=lambda wildcards, output: os.path.dirname(output.run_info),
        blacklist="{path}/{prior_conf}/GLUE/dim:{dim}-alt_dim:{alt_dim}-hidden_depth:{hidden_depth}-hidden_dim:{hidden_dim}-dropout:{dropout}-lam_graph:{lam_graph}-lam_align:{lam_align}-neg_samples:{neg_samples}/seed:{seed}/.blacklist"
    threads: 4
    resources: gpu=1
    shell:
        "timeout {config[timeout]} python -u workflow/scripts/run_GLUE.py "
        "--input-rna {input.rna} --input-atac {input.atac} "
        "-p {input.prior} -d {wildcards.dim} "
        "--alt-dim {wildcards.alt_dim} "
        "--hidden-depth {wildcards.hidden_depth} "
        "--hidden-dim {wildcards.hidden_dim} "
        "--dropout {wildcards.dropout} "
        "--lam-graph {wildcards.lam_graph} "
        "--lam-align {wildcards.lam_align} "
        "--neg-samples {wildcards.neg_samples} "
        "-s {wildcards.seed} "
        "--train-dir {params.train_dir} --random-sleep --require-converge "
        "--output-rna {output.rna_latent} --output-atac {output.atac_latent} "
        "--output-feature {output.feature_latent} --run-info {output.run_info} "
        "> {log} 2>&1 || touch {params.blacklist}"

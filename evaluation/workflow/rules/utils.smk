from utils import default_prior_conf, default_hyperparam_conf

localrules: link_data

rule combine_metrics:
    input:
        lambda wildcards: expand(
            f"{wildcards.path}/{wildcards.dataset}/{wildcards.data_conf}/"
            f"{wildcards.prior_conf}/{wildcards.method}/{wildcards.hyperparam_conf}/"
            f"seed:{wildcards.seed}/{{file}}", file=[
                "run_info.yaml", "cell_integration.yaml",
                *(["feature_consistency.yaml"] if wildcards.method == "GLUE" else [])
            ]
        )
    output:
        "{path}/{dataset}/{data_conf}/{prior_conf}/{method}/{hyperparam_conf}/seed:{seed}/metrics.yaml"
    log:
        "{path}/{dataset}/{data_conf}/{prior_conf}/{method}/{hyperparam_conf}/seed:{seed}/combine_metrics.log"
    threads: 1
    shell:
        "python -u workflow/scripts/combine_metrics.py "
        "-i {input} -o {output} "
        "> {log} 2>&1"

rule feature_consistency:
    input:
        query="{path}/{dataset}/{data_conf}/{prior_conf}/GLUE/{hyperparam_conf}/seed:{seed}/feature_latent.csv",
        ref=lambda wildcards: (
            f"{wildcards.path}/{wildcards.dataset}/original/{default_prior_conf(wildcards.prior_conf)}/GLUE/"
            f"{default_hyperparam_conf(wildcards.hyperparam_conf)}/seed:{wildcards.seed}/feature_latent.csv"
        )
    output:
        "{path}/{dataset}/{data_conf}/{prior_conf}/GLUE/{hyperparam_conf}/seed:{seed}/feature_consistency.yaml"
    log:
        "{path}/{dataset}/{data_conf}/{prior_conf}/GLUE/{hyperparam_conf}/seed:{seed}/feature_consistency.log"
    threads: 1
    shell:
        "python -u workflow/scripts/feature_consistency.py "
        "-q {input.query} -r {input.ref} -o {output} "
        "> {log} 2>&1"

rule cell_integration:
    input:
        rna="{path}/{dataset}/{data_conf}/rna_unirep.h5ad",
        atac="{path}/{dataset}/{data_conf}/atac_unirep.h5ad",
        rna_latent="{path}/{dataset}/{data_conf}/{prior_conf}/{method}/{hyperparam_conf}/seed:{seed}/rna_latent.csv",
        atac_latent="{path}/{dataset}/{data_conf}/{prior_conf}/{method}/{hyperparam_conf}/seed:{seed}/atac_latent.csv"
    output:
        "{path}/{dataset}/{data_conf}/{prior_conf}/{method}/{hyperparam_conf}/seed:{seed}/cell_integration.yaml"
    log:
        "{path}/{dataset}/{data_conf}/{prior_conf}/{method}/{hyperparam_conf}/seed:{seed}/cell_integration.log"
    params:
        paired=lambda wildcards: "-p" if config["dataset"][wildcards.dataset]["paired"] else ""
    threads: 1
    shell:
        "python -u workflow/scripts/cell_integration.py "
        "-d {input.rna} {input.atac} "
        "-l {input.rna_latent} {input.atac_latent} "
        "--cell-type cell_type --domain domain {params.paired} "
        "-o {output} "
        "> {log} 2>&1"

rule rna_unirep:
    input:
        "{path}/rna.h5ad"
    output:
        "{path}/rna_unirep.h5ad"
    log:
        "{path}/rna_unirep.log"
    threads: 1
    shell:
        "python -u workflow/scripts/rna_unirep.py "
        "-i {input} -o {output} "
        "> {log} 2>&1"

rule atac_unirep:
    input:
        "{path}/atac.h5ad"
    output:
        "{path}/atac_unirep.h5ad"
    log:
        "{path}/atac_unirep.log"
    threads: 1
    shell:
        "python -u workflow/scripts/atac_unirep.py "
        "-i {input} -o {output} "
        "> {log} 2>&1"

rule visualize_umap:
    input:
        rna="{path}/rna.h5ad",
        atac="{path}/atac.h5ad",
        rna_umap="{path}/{prior_conf}/{method}/{hyperparam_conf}/seed:{seed}/rna_umap.csv",
        atac_umap="{path}/{prior_conf}/{method}/{hyperparam_conf}/seed:{seed}/atac_umap.csv"
    output:
        "{path}/{prior_conf}/{method}/{hyperparam_conf}/seed:{seed}/{label}.pdf"
    log:
        "{path}/{prior_conf}/{method}/{hyperparam_conf}/seed:{seed}/visualize_umap_{label}.log"
    params:
        title=lambda wildcards: {"cell_type": "'Cell type'", "domain": "'Omics layer'"}[wildcards.label]
    threads: 1
    shell:
        "python -u workflow/scripts/visualize_umap.py "
        "-d {input.rna} {input.atac} "
        "-u {input.rna_umap} {input.atac_umap} "
        "-l {wildcards.label} -t {params.title} "
        "-o {output} "
        "> {log} 2>&1 || touch {output}"  # Discard embeddings that cause segfault in UMAP

rule compute_umap:
    input:
        rna="{path}/rna_latent.csv",
        atac="{path}/atac_latent.csv"
    output:
        rna="{path}/rna_umap.csv",
        atac="{path}/atac_umap.csv"
    log:
        "{path}/compute_umap.log"
    threads: 4
    shell:
        "python -u workflow/scripts/compute_umap.py "
        "-l {input.rna} {input.atac} "
        "-o {output.rna} {output.atac} "
        "> {log} 2>&1 || touch {output.rna} {output.atac}"  # Discard embeddings that cause segfault in UMAP

rule convert_data:
    input:
        rna="{path}/rna.h5ad",
        atac="{path}/atac.h5ad",
        prior="{path}/{prior_conf}/full.graphml.gz"
    output:
        "{path}/{prior_conf}/atac2rna.h5ad"
    log:
        "{path}/{prior_conf}/convert_data.log"
    threads: 1
    shell:
        "python -u workflow/scripts/convert_data.py "
        "--rna {input.rna} --atac {input.atac} "
        "-p {input.prior} -o {output} "
        "> {log} 2>&1"

rule build_prior_graph:
    input:
        rna="{path}/rna.h5ad",
        atac="{path}/atac.h5ad"
    output:
        full="{path}/gene_region:{gene_region}-extend_range:{extend_range}-corrupt_rate:{corrupt_rate}-corrupt_seed:{corrupt_seed}/full.graphml.gz",
        sub="{path}/gene_region:{gene_region}-extend_range:{extend_range}-corrupt_rate:{corrupt_rate}-corrupt_seed:{corrupt_seed}/sub.graphml.gz"
    log:
        "{path}/gene_region:{gene_region}-extend_range:{extend_range}-corrupt_rate:{corrupt_rate}-corrupt_seed:{corrupt_seed}/build_prior_graph.log"
    threads: 1
    shell:
        "python -u workflow/scripts/build_prior_graph.py "
        "--rna {input.rna} --atac {input.atac} "
        "-r {wildcards.gene_region} -e {wildcards.extend_range} "
        "-c {wildcards.corrupt_rate} -s {wildcards.corrupt_seed} "
        "--output-full {output.full} --output-sub {output.sub} "
        "> {log} 2>&1"

rule subsample_data:
    input:
        rna="{path}/{dataset}/original/rna.h5ad",
        atac="{path}/{dataset}/original/atac.h5ad",
        frags2rna="{path}/{dataset}/original/frags2rna.h5ad"
    output:
        rna="{path}/{dataset}/subsample_size:{subsample_size}-subsample_seed:{subsample_seed}/rna.h5ad",
        atac="{path}/{dataset}/subsample_size:{subsample_size}-subsample_seed:{subsample_seed}/atac.h5ad",
        frags2rna="{path}/{dataset}/subsample_size:{subsample_size}-subsample_seed:{subsample_seed}/frags2rna.h5ad"
    log:
        "{path}/{dataset}/subsample_size:{subsample_size}-subsample_seed:{subsample_seed}/subsample_data.log"
    params:
        paired=lambda wildcards: "-p" if config["dataset"][wildcards.dataset]["paired"] else ""
    threads: 1
    shell:
        "python -u workflow/scripts/subsample_data.py "
        "-d {input.rna} {input.atac} {input.frags2rna} "
        "-s {wildcards.subsample_size} {params.paired} "
        "--random-seed {wildcards.subsample_seed} "
        "-o {output.rna} {output.atac} {output.frags2rna} > {log} 2>&1"

rule select_hvg:
    input:
        rna="{path}/{dataset}/original/rna.h5ad",
        atac="{path}/{dataset}/original/atac.h5ad",
        frags2rna="{path}/{dataset}/original/frags2rna.h5ad"
    output:
        rna="{path}/{dataset}/hvg:{hvg}/rna.h5ad",
        atac="{path}/{dataset}/hvg:{hvg}/atac.h5ad",
        frags2rna="{path}/{dataset}/hvg:{hvg}/frags2rna.h5ad"
    log:
        "{path}/{dataset}/hvg:{hvg}/select_hvg.log"
    threads: 1
    shell:
        "python -u workflow/scripts/select_hvg.py "
        "-i {input.rna} -n {wildcards.hvg} -o {output.rna} "
        "> {log} 2>&1 && "
        "ln -frs {input.atac} {output.atac} >> {log} 2>&1 && "
        "ln -frs {input.frags2rna} {output.frags2rna} >> {log} 2>&1"

rule link_data:
    input:
        rna=lambda wildcards: config["dataset"][wildcards.dataset]["rna"],
        atac=lambda wildcards: config["dataset"][wildcards.dataset]["atac"],
        frags2rna=lambda wildcards: config["dataset"][wildcards.dataset]["frags2rna"]
    output:
        rna="{path}/{dataset}/original/rna.h5ad",
        atac="{path}/{dataset}/original/atac.h5ad",
        frags2rna="{path}/{dataset}/original/frags2rna.h5ad"
    log:
        "{path}/{dataset}/original/link_data.log"
    threads: 1
    shell:
        "ln -frs {input.rna} {output.rna} > {log} 2>&1 && "
        "ln -frs {input.atac} {output.atac} >> {log} 2>&1 && "
        "ln -frs {input.frags2rna} {output.frags2rna} >> {log} 2>&1"

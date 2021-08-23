rule metrics:
    input:
        rna="{path}/{dataset}/{subsample_conf}/rna.h5ad",
        atac="{path}/{dataset}/{subsample_conf}/atac.h5ad",
        rna_latent="{path}/{dataset}/{subsample_conf}/{prior_conf}/{method}/{hyperparam_conf}/seed:{seed}/rna_latent.csv",
        atac_latent="{path}/{dataset}/{subsample_conf}/{prior_conf}/{method}/{hyperparam_conf}/seed:{seed}/atac_latent.csv",
        run_info="{path}/{dataset}/{subsample_conf}/{prior_conf}/{method}/{hyperparam_conf}/seed:{seed}/run_info.yaml"
    output:
        "{path}/{dataset}/{subsample_conf}/{prior_conf}/{method}/{hyperparam_conf}/seed:{seed}/metrics.yaml"
    log:
        "{path}/{dataset}/{subsample_conf}/{prior_conf}/{method}/{hyperparam_conf}/seed:{seed}/metrics.log"
    threads: 1
    shell:
        "python -u workflow/scripts/metrics.py "
        "-d {input.rna} {input.atac} "
        "-l {input.rna_latent} {input.atac_latent} "
        "--cell-type cell_type --domain domain -p "
        "-r {input.run_info} -o {output} "
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
        atac="{path}/{dataset}/original/atac.h5ad"
    output:
        rna="{path}/{dataset}/subsample_size:{subsample_size}-subsample_seed:{subsample_seed}/rna.h5ad",
        atac="{path}/{dataset}/subsample_size:{subsample_size}-subsample_seed:{subsample_seed}/atac.h5ad"
    log:
        "{path}/{dataset}/subsample_size:{subsample_size}-subsample_seed:{subsample_seed}/subsample_data.log"
    threads: 1
    shell:
        "python -u workflow/scripts/subsample_data.py "
        "-d {input.rna} {input.atac} "
        "-s {wildcards.subsample_size} -p "
        "--random-seed {wildcards.subsample_seed} "
        "-o {output.rna} {output.atac} > {log} 2>&1"

rule link_data:
    input:
        rna=lambda wildcards: config["dataset"][wildcards.dataset]["rna"],
        atac=lambda wildcards: config["dataset"][wildcards.dataset]["atac"]
    output:
        rna="{path}/{dataset}/original/rna.h5ad",
        atac="{path}/{dataset}/original/atac.h5ad"
    log:
        "{path}/{dataset}/original/link_data.log"
    threads: 1
    shell:
        "ln -frs {input.rna} {output.rna} > {log} 2>&1 && "
        "ln -frs {input.atac} {output.atac} >> {log} 2>&1"

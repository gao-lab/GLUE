#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    source(".Rprofile")
    library(argparse)
    library(Seurat)
    library(harmony)
    library(yaml)
})


parse_args <- function() {
    parser <- ArgumentParser()
    parser$add_argument(
        "--input-rna", dest = "input_rna", type = "character", required = TRUE,
        help = "Path to input RNA dataset (.h5ad)"
    )
    parser$add_argument(
        "--input-atac", dest = "input_atac", type = "character", required = TRUE,
        help = "Path to input ATAC dataset converted to RNA space (.h5ad)"
    )
    parser$add_argument(
        "-s", "--random-seed", dest = "random_seed", type = "integer", default = 0,
        help = "Random seed"
    )
    parser$add_argument(
        "--output-rna", dest = "output_rna", type = "character", required = TRUE,
        help = "Path of output RNA latent file (.csv)"
    )
    parser$add_argument(
        "--output-atac", dest = "output_atac", type = "character", required = TRUE,
        help = "Path of output ATAC latent file (.csv)"
    )
    parser$add_argument(
        "--run-info", dest = "run_info", type = "character", required = TRUE,
        help = "Path of output run info file (.yaml)"
    )
    return(parser$parse_args())
}


main <- function(args) {

    set.seed(args$random_seed)

    cat("[1/4] Reading data...\n")
    rna <- read_h5ad(args$input_rna)
    atac <- read_h5ad(args$input_atac)

    stopifnot(all(rownames(atac$var) == rownames(rna$var)))

    rownames(rna$obs) <- paste(rownames(rna$obs), "RNA", sep = ".")  # Avoid collision
    rownames(rna$X) <- rownames(rna$obs)
    rna.so <- CreateSeuratObject(counts = Matrix::t(rna$X), meta.data = rna$obs)

    rownames(atac$obs) <- paste(rownames(atac$obs), "ATAC", sep = ".")  # Avoid collision
    rownames(atac$X) <- rownames(atac$obs)
    atac.so <- CreateSeuratObject(counts = Matrix::t(atac$X), meta.data = atac$obs)

    hvg <- rownames(rna$var)[rna$var$highly_variable]
    n_cells <- nrow(rna$obs) + nrow(atac$obs)

    combined.so <- merge(rna.so, atac.so)
    rm(rna, atac)
    gc()  # Reduce memory usage

    cat("[2/4] Data preprocessing...\n")
    start_time <- proc.time()
    combined.so <- NormalizeData(combined.so)
    VariableFeatures(combined.so) <- hvg
    combined.so <- ScaleData(combined.so)
    combined.so <- RunPCA(combined.so, seed.use = args$random_seed, verbose = FALSE)

    cat("[3/4] Running Harmony...\n")
    combined.so <- RunHarmony(combined.so, group.by.vars = "domain")
    elapsed_time <- proc.time() - start_time

    cat("[4/4] Saving results...\n")
    combined_latent <- Embeddings(combined.so, reduction="harmony")
    rna_latent <- combined_latent[colnames(rna.so), ]
    rownames(rna_latent) <- gsub("\\.RNA$", "", rownames(rna_latent))
    atac_latent <- combined_latent[colnames(atac.so), ]
    rownames(atac_latent) <- gsub("\\.ATAC$", "", rownames(atac_latent))
    write.table(
        rna_latent, args$output_rna,
        sep = ",", row.names = TRUE, col.names = FALSE, quote = FALSE
    )
    write.table(
        atac_latent, args$output_atac,
        sep = ",", row.names = TRUE, col.names = FALSE, quote = FALSE
    )
    write_yaml(
        list(
            args = args,
            time = elapsed_time["elapsed"],
            n_cells = n_cells
        ), args$run_info
    )
}


main(parse_args())

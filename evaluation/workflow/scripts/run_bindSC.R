#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    source(".Rprofile")
    library(argparse)
    library(bindSC)
    library(Seurat)
    library(Signac)
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
        help = "Path to input ATAC dataset (.h5ad)"
    )
    parser$add_argument(
        "--input-atac2rna", dest = "input_atac2rna", type = "character", required = TRUE,
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
    atac2rna <- read_h5ad(args$input_atac2rna)

    stopifnot(all(rownames(atac2rna$var) == rownames(rna$var)))
    stopifnot(all(rownames(atac2rna$obs) == rownames(atac$obs)))

    rna.so <- CreateSeuratObject(counts = Matrix::t(rna$X), meta.data = rna$obs)
    atac.so <- CreateSeuratObject(counts = Matrix::t(atac$X), meta.data = atac$obs)
    atac2rna.so <- CreateSeuratObject(counts = Matrix::t(atac2rna$X), meta.data = atac$obs)

    hvg <- rownames(rna$var)[rna$var$highly_variable]
    X.clst <- rna$obs$cell_type
    Y.clst <- atac$obs$cell_type
    n_cells <- nrow(rna$obs) + nrow(atac$obs)

    rm(rna, atac, atac2rna)
    gc()  # Reduce memory usage

    cat("[2/4] Data preprocessing...\n")
    start_time <- proc.time()
    rna.so <- NormalizeData(rna.so)
    atac2rna.so <- NormalizeData(atac2rna.so)

    X <- GetAssayData(rna.so)[hvg, ]
    Z0 <- GetAssayData(atac2rna.so)[hvg, ]
    out <- dimReduce(dt1 = X, dt2 = Z0, K = 30)
    X <- t(out$dt1)
    Z0 <- t(out$dt2)

    atac.so <- RunTFIDF(atac.so)
    atac.so <- FindTopFeatures(atac.so, min.cutoff = "q0")
    atac.so <- RunSVD(atac.so, n = 50)

    Y <- t(Embeddings(atac.so, reduction = "lsi"))

    rm(rna.so, atac.so, atac2rna.so)
    gc()  # Reduce memory usage

    cat("[3/4] Running bindSC...\n")
    res <- BiCCA(
        X = X, Z0 = Z0, Y = Y, X.clst = X.clst, Y.clst = Y.clst,
        alpha = 0.5, lambda = 0.5, K = 15,
        temp.path = dirname(args$run_info), save = FALSE
    )
    elapsed_time <- proc.time() - start_time

    cat("[4/4] Saving results...\n")
    write.table(
        res$u, args$output_rna,
        sep = ",", row.names = TRUE, col.names = FALSE, quote = FALSE
    )
    write.table(
        res$r, args$output_atac,
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

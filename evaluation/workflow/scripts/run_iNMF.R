#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    source(".Rprofile")
    library(argparse)
    library(rliger)
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

    rownames(rna$obs) <- paste(rownames(rna$obs), "RNA", sep = ".")  # Avoid collision
    rownames(rna$X) <- rownames(rna$obs)

    rownames(atac$obs) <- paste(rownames(atac$obs), "ATAC", sep = ".")  # Avoid collision
    rownames(atac$X) <- rownames(atac$obs)

    int.liger <- createLiger(list(
        atac = Matrix::t(atac$X),
        rna = Matrix::t(rna$X)
    ))

    hvg <- rownames(rna$var)[rna$var$highly_variable]
    stopifnot(all(hvg %in% rownames(rna$var)))
    stopifnot(all(hvg %in% rownames(atac$var)))
    hvg <- intersect(hvg, rownames(int.liger@raw.data$rna))  # Because of feature filtering in createLiger
    hvg <- intersect(hvg, rownames(int.liger@raw.data$atac))  # Because of feature filtering in createLiger
    rna_cells <- rownames(rna$obs)
    atac_cells <- rownames(atac$obs)
    n_cells <- nrow(rna$obs) + nrow(atac$obs)
    min_cells <- min(nrow(rna$obs), nrow(atac$obs))

    rm(rna, atac)
    gc()  # Reduce memory usage

    cat("[2/4] Data preprocessing...\n")
    start_time <- proc.time()
    int.liger <- normalize(int.liger)

    int.liger@var.genes <- hvg
    int.liger <- scaleNotCenter(int.liger)

    cat("[3/4] Running iNMF...\n")
    int.liger <- online_iNMF(int.liger, k = 20, miniBatch_size = min(5000, min_cells), seed = args$random_seed)
    int.liger <- quantile_norm(int.liger, rand.seed = args$random_seed)
    elapsed_time <- proc.time() - start_time

    cat("[4/4] Saving results...\n")
    combined_latent <- int.liger@H.norm
    missing_cells <- setdiff(
        union(rna_cells, atac_cells),
        rownames(combined_latent)
    )  # Because of cell filtering in scaleNotCenter
    combined_latent <- rbind(combined_latent, matrix(
        nrow = length(missing_cells),
        ncol = ncol(combined_latent),
        dimnames = list(missing_cells, colnames(combined_latent))
    ))  # Fill with NA
    rna_latent <- combined_latent[rna_cells, ]
    rownames(rna_latent) <- gsub("\\.RNA$", "", rownames(rna_latent))
    atac_latent <- combined_latent[atac_cells, ]
    rownames(atac_latent) <- gsub("\\.ATAC$", "", rownames(atac_latent))
    write.table(
        rna_latent, args$output_rna,
        sep = ",", row.names = TRUE, col.names = FALSE, quote = FALSE, na = ""
    )
    write.table(
        atac_latent, args$output_atac,
        sep = ",", row.names = TRUE, col.names = FALSE, quote = FALSE, na = ""
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

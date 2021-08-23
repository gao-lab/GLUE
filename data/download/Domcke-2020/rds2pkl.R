#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    source("../../../.Rprofile", chdir = TRUE)
    library(argparse)
    library(reticulate)
    library(Seurat)
})


parse_args <- function() {
    parser <- ArgumentParser()
    parser$add_argument(
        "-i", "--input", dest = "input", type = "character", required = TRUE,
        help = "Path to the input RDS file (.RDS)"
    )
    parser$add_argument(
        "-o", "--output", dest = "output", type = "character", required = TRUE,
        help = "Path to the output pickle file (.pkl.gz)"
    )
    return(parser$parse_args())
}


main <- function(args) {
    pickle <- import("pickle", convert = FALSE)
    gzip <- import("gzip", convert = FALSE)
    dict <- import_builtins()$dict

    so <- readRDS(args$input)
    mat <- GetAssayData(so, assay = "peaks")
    X <- r_to_py(mat)$T
    obs <- r_to_py(so@meta.data)
    var_names <- r_to_py(rownames(mat))

    f <- gzip$open(args$output, "wb")
    pickle$dump(dict(X = X, obs = obs, var_names = var_names), f)
    f$close()
}


main(parse_args())

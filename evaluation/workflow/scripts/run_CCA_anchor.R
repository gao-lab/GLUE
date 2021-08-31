#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    source(".Rprofile")
    library(argparse)
    library(Seurat, lib.loc = "custom")
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

    rownames(rna$obs) <- paste(rownames(rna$obs), "RNA", sep = ".")  # Avoid collision
    rownames(rna$X) <- rownames(rna$obs)
    rna.so <- CreateSeuratObject(counts = Matrix::t(rna$X), assay = "RNA", meta.data = rna$obs)

    rownames(atac$obs) <- paste(rownames(atac$obs), "ATAC", sep = ".")  # Avoid collision
    rownames(atac$X) <- rownames(atac$obs)
    atac.so <- CreateSeuratObject(counts = Matrix::t(atac$X), assay = "ATAC", meta.data = atac$obs)

    rownames(atac2rna$obs) <- paste(rownames(atac2rna$obs), "ATAC", sep = ".")  # Avoid collision
    rownames(atac2rna$X) <- rownames(atac2rna$obs)
    atac.so[["ACTIVITY"]] <- CreateAssayObject(counts = Matrix::t(atac2rna$X))

    hvg <- rownames(rna$var)[rna$var$highly_variable]
    n_cells <- nrow(rna$obs) + nrow(atac$obs)

    rm(rna, atac, atac2rna)
    gc()  # Reduce memory usage

    cat("[2/4] Data preprocessing...\n")
    start_time <- proc.time()
    rna.so <- NormalizeData(rna.so)
    VariableFeatures(rna.so) <- hvg
    rna.so <- ScaleData(rna.so)
    rna.so <- RunPCA(rna.so, seed.use = args$random_seed, verbose = FALSE)

    DefaultAssay(atac.so) <- "ACTIVITY"
    VariableFeatures(atac.so) <- hvg
    atac.so <- NormalizeData(atac.so)
    atac.so <- ScaleData(atac.so)

    DefaultAssay(atac.so) <- "ATAC"
    atac.so <- RunTFIDF(atac.so)
    atac.so <- FindTopFeatures(atac.so, min.cutoff = "q0")
    atac.so <- RunSVD(atac.so)

    cat("[3/4] Running CCA anchor...\n")
    transfer.anchors <- FindTransferAnchors(
        reference = rna.so, query = atac.so,
        features = VariableFeatures(rna.so),
        reference.assay = "RNA", query.assay = "ACTIVITY",
        reduction = "cca", seed.use = args$random_seed
    )
    refdata <- GetAssayData(
        rna.so, assay = "RNA", slot = "data"
    )[VariableFeatures(rna.so), ]
    imputation <- TransferData(
        anchorset = transfer.anchors, refdata = refdata,
        weight.reduction = atac.so[["lsi"]], dims = 1:30
    )
    atac.so[["RNA"]] <- imputation
    coembed.so <- merge(x = rna.so, y = atac.so)
    coembed.so <- ScaleData(
        coembed.so, features = VariableFeatures(rna.so), do.scale = FALSE
    )
    coembed.so <- RunPCA(
        coembed.so, features = VariableFeatures(rna.so),
        seed.use = args$random_seed, verbose = FALSE
    )
    elapsed_time <- proc.time() - start_time

    cat("[4/4] Saving results...\n")
    combined_latent <- Embeddings(coembed.so)
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

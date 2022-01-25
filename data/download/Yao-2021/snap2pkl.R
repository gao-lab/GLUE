suppressPackageStartupMessages({
    source("../../../.Rprofile", chdir = TRUE)
    library(argparse)
    library(reticulate)
    library(SnapATAC)
})


parse_args <- function() {
    parser <- ArgumentParser()
    parser$add_argument(
        "-i", "--input", dest = "input", type = "character", required = TRUE,
        help = "Path to the input snap file (.snap)"
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

    snap <- createSnap(args$input, basename(args$input))
    snap <- addPmatToSnap(snap)
    X <- r_to_py(snap@pmat)$tocsr()
    obs_names <- r_to_py(snap@barcode)
    var_names <- r_to_py(snap@peak$name)

    f <- gzip$open(args$output, "wb")
    pickle$dump(dict(X = X, obs_names = obs_names, var_names = var_names), f)
    f$close()
}


main(parse_args())

#### -- Packrat Autoloader (version 0.5.0) -- ####
source("packrat/init.R")
#### -- End Packrat Autoloader -- ####

options(bitmapType = "cairo")


read_h5ad <- function(filename) {
    builtins <- reticulate::import_builtins()
    anndata <- reticulate::import("anndata", convert = FALSE)

    Mapping <- reticulate::import("typing")$Mapping
    DataFrame <- reticulate::import("pandas")$DataFrame
    issparse <- reticulate::import("scipy.sparse")$issparse
    isinstance <- builtins$isinstance

    adata <- anndata$read_h5ad(filename)

    .convert <- function(obj) {
        if (!isinstance(obj, Mapping) || isinstance(obj, DataFrame)) {
            return(reticulate::py_to_r(obj))
        }
        ret <- list()
        for (item in builtins$list(obj$keys())) {
            ret[[item]] <- .convert(obj[[item]])
        }
        return(ret)
    }

    if (issparse(adata$X)) {
        X <- .convert(adata$X$tocsc())
    } else {
        X <- .convert(adata$X)
    }
    layers <- .convert(adata$layers)
    obs <- .convert(adata$obs)
    var <- .convert(adata$var)
    obsm <- .convert(adata$obsm)
    varm <- .convert(adata$varm)
    obsp <- .convert(adata$obsp)
    varp <- .convert(adata$varp)
    uns <- .convert(adata$uns)
    rownames(X) <- rownames(obs)
    colnames(X) <- rownames(var)

    return(list(
        X = X, layers = layers,
        obs = obs, var = var,
        obsm = obsm, varm = varm,
        obsp = obsp, varp = varp,
        uns = uns
    ))
}


safe_sd <- function(x) {
    if (length(x) == 1)
        return(0.0)
    return(sd(x))
}


minmax_scale <- function(x, x.min, x.max) {
    if (missing(x.min)) x.min <- min(x)
    if (missing(x.max)) x.max <- max(x)
    return((x - x.min) / (x.max - x.min))
}


ggplot_theme <- function(...) {
    return(ggplot2::theme(
        text = ggplot2::element_text(family = "ArialMT"),
        plot.background = ggplot2::element_blank(),
        panel.grid.major = ggplot2::element_line(color = "#EEEEEE", linetype = "longdash"),
        panel.grid.minor = ggplot2::element_blank(),
        panel.background = ggplot2::element_rect(fill = "#FFFFFF"),
        legend.background = ggplot2::element_blank(),
        legend.box.background = ggplot2::element_blank(),
        axis.line = ggplot2::element_line(color = "#000000"),
        ...
    ))
}


ggplot_save <- function(filename, ...) {
    ggplot2::ggsave(filename, ..., dpi = 600, bg = "transparent")
}

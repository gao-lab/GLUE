suppressPackageStartupMessages({
    source(".Rprofile")
    library(dplyr)
    library(ggplot2)
    library(ggpubr)
    library(rlang)
    library(yaml)
})
set.seed(0)


#-------------------------------- Preparations ---------------------------------

display <- read_yaml("config/display.yaml")
params <- snakemake@config$method$GLUE
defaults <- lapply(params, function(x) x$default)

ref <- read.csv(snakemake@input[[1]])
min_mean_average_precision <- min(ref$mean_average_precision)
max_mean_average_precision <- max(ref$mean_average_precision)
min_avg_silhouette_width <- min(ref$avg_silhouette_width)
max_avg_silhouette_width <- max(ref$avg_silhouette_width)
min_neighbor_conservation <- min(ref$neighbor_conservation)
max_neighbor_conservation <- max(ref$neighbor_conservation)
min_seurat_alignment_score <- min(ref$seurat_alignment_score)
max_seurat_alignment_score <- max(ref$seurat_alignment_score)
min_avg_silhouette_width_batch <- min(ref$avg_silhouette_width_batch)
max_avg_silhouette_width_batch <- max(ref$avg_silhouette_width_batch)
min_graph_connectivity <- min(ref$graph_connectivity)
max_graph_connectivity <- max(ref$graph_connectivity)

df <- read.csv(snakemake@input[[2]]) %>%
    mutate(
        dataset = factor(dataset, levels = names(display[["dataset"]]), labels = display[["dataset"]]),
        bio = (
            minmax_scale(mean_average_precision, min_mean_average_precision, max_mean_average_precision) +
            minmax_scale(avg_silhouette_width, min_avg_silhouette_width, max_avg_silhouette_width) +
            minmax_scale(neighbor_conservation, min_neighbor_conservation, max_neighbor_conservation)
        ) / 3,
        int = (
            minmax_scale(seurat_alignment_score, min_seurat_alignment_score, max_seurat_alignment_score) +
            minmax_scale(avg_silhouette_width_batch, min_avg_silhouette_width_batch, max_avg_silhouette_width_batch) +
            minmax_scale(graph_connectivity, min_graph_connectivity, max_graph_connectivity)
        ) / 3,
        overall = 0.6 * bio + 0.4 * int
    ) %>%
    as.data.frame()

df_list <- list()
for (param in names(params)) {
    df_filter <- duplicate(df)
    for (constant in names(params)) {
        if (param == constant) next
        df_filter <- df_filter %>%
            filter(!!as.symbol(constant) == defaults[[constant]])
    }
    df_filter <- df_filter %>%
        select(
            dataset, !!as.symbol(param), foscttm,
            mean_average_precision, seurat_alignment_score,
            avg_silhouette_width, avg_silhouette_width_batch,
            neighbor_conservation, graph_connectivity,
            bio, int, overall,
            feature_consistency
        ) %>%
        rename(value = !!as.symbol(param))
    df_list[[param]] <- df_filter
}

df <- bind_rows(df_list, .id = "param") %>%
    mutate(
        param = factor(param, levels = names(display[["param"]]), labels = display[["param"]]),
        value = factor(as.character(value), as.character(sort(unique(value))))
    ) %>%
    group_by(dataset, param, value) %>%
    summarise(
        foscttm_sd = safe_sd(foscttm),
        foscttm = mean(foscttm),
        mean_average_precision_sd = safe_sd(mean_average_precision),
        mean_average_precision = mean(mean_average_precision),
        seurat_alignment_score_sd = safe_sd(seurat_alignment_score),
        seurat_alignment_score = mean(seurat_alignment_score),
        avg_silhouette_width_sd = safe_sd(avg_silhouette_width),
        avg_silhouette_width = mean(avg_silhouette_width),
        avg_silhouette_width_batch_sd = safe_sd(avg_silhouette_width_batch),
        avg_silhouette_width_batch = mean(avg_silhouette_width_batch),
        neighbor_conservation_sd = safe_sd(neighbor_conservation),
        neighbor_conservation = mean(neighbor_conservation),
        graph_connectivity_sd = safe_sd(graph_connectivity),
        graph_connectivity = mean(graph_connectivity),
        bio_sd = safe_sd(bio),
        bio = mean(bio),
        int_sd = safe_sd(int),
        int = mean(int),
        overall_sd = safe_sd(overall),
        overall = mean(overall),
        feature_consistency_sd = safe_sd(feature_consistency),
        feature_consistency = mean(feature_consistency)
    ) %>%
    as.data.frame()

paired_datasets <- names(Filter(function(x) x$paired, snakemake@config[["dataset"]]))
paired_datasets <- Map(function(x) display$dataset[[x]], paired_datasets)


#---------------------------------- Plotting -----------------------------------

gp <- ggplot(data = df %>% filter(dataset %in% paired_datasets), mapping = aes(
    x = value, y = foscttm,
    ymax = foscttm + foscttm_sd,
    ymin = foscttm - foscttm_sd,
    color = dataset, group = dataset
)) +
    geom_point() + geom_line() + geom_errorbar(width = 0.1) +
    facet_wrap(~ param, scales = "free_x") +
    scale_x_discrete(name = "Hyperparameter value") +
    scale_y_continuous(name = "FOSCTTM") +
    scale_color_manual(name = "Dataset", values = unlist(display[["palette"]])) +
    guides(color = FALSE) +
    ggplot_theme()
ggplot_save(snakemake@output[["foscttm"]], gp, width = 6, height = 6.5)

gp <- ggplot(data = df, mapping = aes(
    x = value, y = mean_average_precision,
    ymax = mean_average_precision + mean_average_precision_sd,
    ymin = mean_average_precision - mean_average_precision_sd,
    color = dataset, group = dataset
)) +
    geom_point() + geom_line() + geom_errorbar(width = 0.1) +
    facet_wrap(~ param, scales = "free_x") +
    scale_x_discrete(name = "Hyperparameter value") +
    scale_y_continuous(name = "Mean average precision") +
    scale_color_manual(name = "Dataset", values = unlist(display[["palette"]])) +
    guides(color = FALSE) +
    ggplot_theme()
ggplot_save(snakemake@output[["map"]], gp, width = 6, height = 6.5)

gp <- ggplot(data = df, mapping = aes(
    x = value, y = seurat_alignment_score,
    ymax = seurat_alignment_score + seurat_alignment_score_sd,
    ymin = seurat_alignment_score - seurat_alignment_score_sd,
    color = dataset, group = dataset
)) +
    geom_point() + geom_line() + geom_errorbar(width = 0.1) +
    facet_wrap(~ param, scales = "free_x") +
    scale_x_discrete(name = "Hyperparameter value") +
    scale_y_continuous(name = "Seurat alignment score") +
    scale_color_manual(name = "Dataset", values = unlist(display[["palette"]])) +
    guides(color = FALSE) +
    ggplot_theme()
ggplot_save(snakemake@output[["sas"]], gp, width = 6, height = 6.5)

gp <- ggplot(data = df, mapping = aes(
    x = value, y = avg_silhouette_width,
    ymax = avg_silhouette_width + avg_silhouette_width_sd,
    ymin = avg_silhouette_width - avg_silhouette_width_sd,
    color = dataset, group = dataset
)) +
    geom_point() + geom_line() + geom_errorbar(width = 0.1) +
    facet_wrap(~ param, scales = "free_x") +
    scale_x_discrete(name = "Hyperparameter value") +
    scale_y_continuous(name = "Average silhouette width (cell type)") +
    scale_color_manual(name = "Dataset", values = unlist(display[["palette"]])) +
    guides(color = FALSE) +
    ggplot_theme()
ggplot_save(snakemake@output[["asw"]], gp, width = 6, height = 6.5)

gp <- ggplot(data = df, mapping = aes(
    x = value, y = avg_silhouette_width_batch,
    ymax = avg_silhouette_width_batch + avg_silhouette_width_batch_sd,
    ymin = avg_silhouette_width_batch - avg_silhouette_width_batch_sd,
    color = dataset, group = dataset
)) +
    geom_point() + geom_line() + geom_errorbar(width = 0.1) +
    facet_wrap(~ param, scales = "free_x") +
    scale_x_discrete(name = "Hyperparameter value") +
    scale_y_continuous(name = "Average silhouette width (omics layer)") +
    scale_color_manual(name = "Dataset", values = unlist(display[["palette"]])) +
    guides(color = FALSE) +
    ggplot_theme()
ggplot_save(snakemake@output[["aswb"]], gp, width = 6, height = 6.5)

gp <- ggplot(data = df, mapping = aes(
    x = value, y = neighbor_conservation,
    ymax = neighbor_conservation + neighbor_conservation_sd,
    ymin = neighbor_conservation - neighbor_conservation_sd,
    color = dataset, group = dataset
)) +
    geom_point() + geom_line() + geom_errorbar(width = 0.1) +
    facet_wrap(~ param, scales = "free_x") +
    scale_x_discrete(name = "Hyperparameter value") +
    scale_y_continuous(name = "Neighbor conservation") +
    scale_color_manual(name = "Dataset", values = unlist(display[["palette"]])) +
    guides(color = FALSE) +
    ggplot_theme()
ggplot_save(snakemake@output[["nc"]], gp, width = 6, height = 6.5)

gp <- ggplot(data = df, mapping = aes(
    x = value, y = graph_connectivity,
    ymax = graph_connectivity + graph_connectivity_sd,
    ymin = graph_connectivity - graph_connectivity_sd,
    color = dataset, group = dataset
)) +
    geom_point() + geom_line() + geom_errorbar(width = 0.1) +
    facet_wrap(~ param, scales = "free_x") +
    scale_x_discrete(name = "Hyperparameter value") +
    scale_y_continuous(name = "Graph connectivity") +
    scale_color_manual(name = "Dataset", values = unlist(display[["palette"]])) +
    guides(color = FALSE) +
    ggplot_theme()
ggplot_save(snakemake@output[["gc"]], gp, width = 6, height = 6.5)

gp <- ggplot(data = df, mapping = aes(
    x = value, y = bio,
    ymax = bio + bio_sd,
    ymin = bio - bio_sd,
    color = dataset, group = dataset
)) +
    geom_point() + geom_line() + geom_errorbar(width = 0.1) +
    facet_wrap(~ param, scales = "free_x") +
    scale_x_discrete(name = "Hyperparameter value") +
    scale_y_continuous(name = "Biology conservation") +
    scale_color_manual(name = "Dataset", values = unlist(display[["palette"]])) +
    guides(color = FALSE) +
    ggplot_theme()
ggplot_save(snakemake@output[["bio"]], gp, width = 6, height = 6.5)

gp <- ggplot(data = df, mapping = aes(
    x = value, y = int,
    ymax = int + int_sd,
    ymin = int - int_sd,
    color = dataset, group = dataset
)) +
    geom_point() + geom_line() + geom_errorbar(width = 0.1) +
    facet_wrap(~ param, scales = "free_x") +
    scale_x_discrete(name = "Hyperparameter value") +
    scale_y_continuous(name = "Omics mixing") +
    scale_color_manual(name = "Dataset", values = unlist(display[["palette"]])) +
    guides(color = FALSE) +
    ggplot_theme()
ggplot_save(snakemake@output[["int"]], gp, width = 6, height = 6.5)

gp <- ggplot(data = df, mapping = aes(
    x = value, y = overall,
    ymax = overall + overall_sd,
    ymin = overall - overall_sd,
    color = dataset, group = dataset
)) +
    geom_point() + geom_line() + geom_errorbar(width = 0.1) +
    facet_wrap(~ param, scales = "free_x") +
    scale_x_discrete(name = "Hyperparameter value") +
    scale_y_continuous(name = "Overall integration score") +
    scale_color_manual(name = "Dataset", values = unlist(display[["palette"]])) +
    guides(color = FALSE) +
    ggplot_theme()
ggplot_save(snakemake@output[["overall"]], gp, width = 6, height = 6.5)

gp <- ggplot(data = df, mapping = aes(
    x = value, y = feature_consistency,
    ymax = feature_consistency + feature_consistency_sd,
    ymin = feature_consistency - feature_consistency_sd,
    color = dataset, group = dataset
)) +
    geom_point() + geom_line() + geom_errorbar(width = 0.1) +
    facet_wrap(~ param, scales = "free_x") +
    scale_x_discrete(name = "Hyperparameter value") +
    scale_y_continuous(name = "Feature consistency") +
    scale_color_manual(name = "Dataset", values = unlist(display[["palette"]])) +
    guides(color = FALSE) +
    ggplot_theme()
ggplot_save(snakemake@output[["fc"]], gp, width = 6, height = 6.5)

gp <- ggplot(
    data = df,
    mapping = aes(x = dataset, y = overall, fill = dataset)
) +
    geom_bar(stat = "identity") +
    scale_fill_manual(name = "Dataset", values = unlist(display[["palette"]])) +
    guides(fill = guide_legend(nrow = 1)) +
    ggplot_theme(legend.direction = "horizontal")
leg <- as_ggplot(get_legend(gp))
ggsave(snakemake@output[["legend_h"]], leg, width = 7, height = 1)

gp <- ggplot(
    data = df,
    mapping = aes(x = dataset, y = overall, fill = dataset)
) +
    geom_bar(stat = "identity") +
    scale_fill_manual(name = "Dataset", values = unlist(display[["palette"]])) +
    guides(fill = guide_legend(ncol = 1)) +
    ggplot_theme()
leg <- as_ggplot(get_legend(gp))
ggsave(snakemake@output[["legend_v"]], leg, width = 2, height = 2)

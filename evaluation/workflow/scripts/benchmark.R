suppressPackageStartupMessages({
    source(".Rprofile")
    library(plyr)
    library(dplyr)
    library(ggplot2)
    library(ggpubr)
    library(yaml)
})
set.seed(0)


#-------------------------------- Preparations ---------------------------------

display <- read_yaml("config/display.yaml")

if (grepl("noprep", snakemake@input[[1]])) {
    display[["method"]][["GLUE"]] <- "GLUE (no PCA/LSI)"
    display[["palette"]][["GLUE (no PCA/LSI)"]] <- display[["palette"]][["GLUE"]]
}

df <- read.csv(snakemake@input[[1]]) %>%
    mutate(
        dataset = factor(dataset, levels = names(display[["dataset"]]), labels = display[["dataset"]]),
        method = factor(method, levels = names(display[["method"]]), labels = display[["method"]]),
        bio = (
            minmax_scale(mean_average_precision) +
            minmax_scale(avg_silhouette_width) +
            minmax_scale(neighbor_conservation)
        ) / 3,
        int = (
            minmax_scale(seurat_alignment_score) +
            minmax_scale(avg_silhouette_width_batch) +
            minmax_scale(graph_connectivity)
        ) / 3,
        overall = 0.6 * bio + 0.4 * int
    ) %>%
    as.data.frame()

df_summarise <- df %>%
    group_by(dataset, method) %>%
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
        graph_connectivity_sd = safe_sd(graph_connectivity),
        graph_connectivity = mean(graph_connectivity),
        neighbor_conservation_sd = safe_sd(neighbor_conservation),
        neighbor_conservation = mean(neighbor_conservation),
        bio_sd = safe_sd(bio),
        bio = mean(bio),
        int_sd = safe_sd(int),
        int = mean(int),
        overall_sd = safe_sd(overall),
        overall = mean(overall),
        n_seeds = length(seed)
    ) %>%
    as.data.frame()

for (dataset in levels(df_summarise$dataset)) {
    missing_methods <- setdiff(
        levels(df_summarise$method),
        df_summarise$method[df_summarise$dataset == dataset]
    )
    if (length(missing_methods > 0)) {
        placeholder <- data.frame(
            dataset = factor(dataset, levels = levels(df_summarise$dataset)),
            method = factor(missing_methods, levels = levels(df_summarise$method)),
            annotation = "N.A."
        )
        df <- bind_rows(df, placeholder)
        df_summarise <- bind_rows(df_summarise, placeholder)
    }
}

df <- df %>%
    merge(df_summarise[, c("dataset", "method", "n_seeds")]) %>%
    mutate(alpha = pmin(1, 4 / n_seeds)) %>%
    arrange(method)
df_summarise <- df_summarise %>% arrange(method)

paired_datasets <- names(Filter(function(x) x$paired, snakemake@config[["dataset"]]))
unpaired_datasets <- names(Filter(function(x) !x$paired, snakemake@config[["dataset"]]))
paired_datasets <- Map(function(x) display$dataset[[x]], paired_datasets)
unpaired_datasets <- Map(function(x) display$dataset[[x]], unpaired_datasets)


#---------------------------------- Plotting -----------------------------------

gp <- ggplot(
    data = df_summarise %>% filter(dataset %in% paired_datasets),
    mapping = aes(x = dataset, y = foscttm, fill = method)
) +
    geom_bar(stat = "identity", width = 0.8, position = position_dodge2(width = 0.8)) +
    geom_point(
        data = df %>% filter(dataset %in% paired_datasets), size = 0.2,
        position = position_jitterdodge(jitter.width = 0.3, dodge.width = 0.8),
        show.legend = FALSE
    ) +
    geom_errorbar(
        mapping = aes(ymin = foscttm - foscttm_sd, ymax = foscttm + foscttm_sd),
        width = 0.2, position = position_dodge(width = 0.8)
    ) +
    geom_text(
        mapping = aes(label = annotation), y = 0.035, vjust = 0.5, size = 2.3,
        angle = 90, position = position_dodge2(width = 0.8), fontface = "bold"
    ) +
    scale_fill_manual(name = "Method", values = unlist(display[["palette"]])) +
    scale_x_discrete(name = "Dataset") +
    scale_y_continuous(name = "FOSCTTM") +
    guides(fill = FALSE) +
    ggplot_theme()
ggplot_save(snakemake@output[["foscttm"]], gp, width = 5, height = 3)

gp <- ggplot(
    data = df_summarise,
    mapping = aes(x = mean_average_precision, y = seurat_alignment_score, color = method)
) +
    geom_errorbar(mapping = aes(
        ymin = seurat_alignment_score - seurat_alignment_score_sd,
        ymax = seurat_alignment_score + seurat_alignment_score_sd
    ), width = 0) +
    geom_errorbarh(mapping = aes(
        xmin = mean_average_precision - mean_average_precision_sd,
        xmax = mean_average_precision + mean_average_precision_sd
    ), height = 0) +
    geom_point(data = df, mapping = aes(alpha = alpha)) +
    facet_wrap(~dataset) +
    scale_alpha_continuous(limits = c(0, 1), range = c(0, 1)) +
    scale_color_manual(name = "Method", values = unlist(display[["palette"]])) +
    scale_x_continuous(name = "Mean average precision") +
    scale_y_continuous(name = "Seurat alignment score") +
    guides(color = FALSE, alpha = FALSE) +
    ggplot_theme()
ggplot_save(snakemake@output[["map_vs_sas"]], gp, width = 5.5, height = 4)

hull <- ddply(
    df_summarise, "method",
    function(df) df %>%
        filter(!is.na(mean_average_precision) & !is.na(seurat_alignment_score)) %>%
        slice(chull(mean_average_precision, seurat_alignment_score))
) %>% arrange(method)
gp <- ggplot(mapping = aes(
    x = mean_average_precision, y = seurat_alignment_score,
    color = method, fill = method
)) +
    geom_polygon(data = hull, alpha = 0.1, linetype = "dashed") +
    geom_point(data = df_summarise, mapping = aes(shape = dataset), size = 1.5) +
    scale_color_manual(name = "Method", values = unlist(display[["palette"]])) +
    scale_fill_manual(name = "Method", values = unlist(display[["palette"]])) +
    scale_shape_manual(name = "Dataset", values = unlist(display[["shape"]])) +
    scale_x_continuous(name = "Mean average precision") +
    scale_y_continuous(name = "Seurat alignment score") +
    guides(color = FALSE, fill = FALSE, shape = FALSE) +
    ggplot_theme()
ggsave(snakemake@output[["map_vs_sas_hull"]], gp, width = 3, height = 3)

gp <- ggplot(
    data = df_summarise,
    mapping = aes(x = avg_silhouette_width, y = avg_silhouette_width_batch, color = method)
) +
    geom_errorbar(mapping = aes(
        ymin = avg_silhouette_width_batch - avg_silhouette_width_batch_sd,
        ymax = avg_silhouette_width_batch + avg_silhouette_width_batch_sd
    ), width = 0) +
    geom_errorbarh(mapping = aes(
        xmin = avg_silhouette_width - avg_silhouette_width_sd,
        xmax = avg_silhouette_width + avg_silhouette_width_sd
    ), height = 0) +
    geom_point(data = df, mapping = aes(alpha = alpha)) +
    facet_wrap(~dataset) +
    scale_alpha_continuous(limits = c(0, 1), range = c(0, 1)) +
    scale_color_manual(name = "Method", values = unlist(display[["palette"]])) +
    scale_x_continuous(name = "Average silhouette width (cell type)") +
    scale_y_continuous(name = "Average silhouette width (omics layer)") +
    guides(color = FALSE, alpha = FALSE) +
    ggplot_theme()
ggplot_save(snakemake@output[["asw_vs_aswb"]], gp, width = 5.5, height = 4)

hull <- ddply(
    df_summarise, "method",
    function(df) df %>%
        filter(!is.na(avg_silhouette_width) & !is.na(avg_silhouette_width_batch)) %>%
        slice(chull(avg_silhouette_width, avg_silhouette_width_batch))
) %>% arrange(method)
gp <- ggplot(mapping = aes(
    x = avg_silhouette_width, y = avg_silhouette_width_batch,
    color = method, fill = method
)) +
    geom_polygon(data = hull, alpha = 0.1, linetype = "dashed") +
    geom_point(data = df_summarise, mapping = aes(shape = dataset), size = 1.5) +
    scale_color_manual(name = "Method", values = unlist(display[["palette"]])) +
    scale_fill_manual(name = "Method", values = unlist(display[["palette"]])) +
    scale_shape_manual(name = "Dataset", values = unlist(display[["shape"]])) +
    scale_x_continuous(name = "Average silhouette width (cell type)") +
    scale_y_continuous(name = "Average silhouette width (omics layer)") +
    guides(color = FALSE, fill = FALSE, shape = FALSE) +
    ggplot_theme()
ggsave(snakemake@output[["asw_vs_aswb_hull"]], gp, width = 3, height = 3)

gp <- ggplot(
    data = df_summarise,
    mapping = aes(x = neighbor_conservation, y = graph_connectivity, color = method)
) +
    geom_errorbar(mapping = aes(
        ymin = graph_connectivity - graph_connectivity_sd,
        ymax = graph_connectivity + graph_connectivity_sd
    ), width = 0) +
    geom_errorbarh(mapping = aes(
        xmin = neighbor_conservation - neighbor_conservation_sd,
        xmax = neighbor_conservation + neighbor_conservation_sd
    ), height = 0) +
    geom_point(data = df, mapping = aes(alpha = alpha)) +
    facet_wrap(~dataset) +
    scale_alpha_continuous(limits = c(0, 1), range = c(0, 1)) +
    scale_color_manual(name = "Method", values = unlist(display[["palette"]])) +
    scale_x_continuous(name = "Neighbor conservation") +
    scale_y_continuous(name = "Graph connectivity") +
    guides(color = FALSE, alpha = FALSE) +
    ggplot_theme()
ggplot_save(snakemake@output[["nc_vs_gc"]], gp, width = 5.5, height = 4)

hull <- ddply(
    df_summarise, "method",
    function(df) df %>%
        filter(!is.na(neighbor_conservation) & !is.na(graph_connectivity)) %>%
        slice(chull(neighbor_conservation, graph_connectivity))
) %>% arrange(method)
gp <- ggplot(mapping = aes(
    x = neighbor_conservation, y = graph_connectivity,
    color = method, fill = method
)) +
    geom_polygon(data = hull, alpha = 0.1, linetype="dashed") +
    geom_point(data = df_summarise, mapping = aes(shape = dataset), size = 1.5) +
    scale_color_manual(name = "Method", values = unlist(display[["palette"]])) +
    scale_fill_manual(name = "Method", values = unlist(display[["palette"]])) +
    scale_shape_manual(name = "Dataset", values = unlist(display[["shape"]])) +
    scale_x_continuous(name = "Neighbor conservation") +
    scale_y_continuous(name = "Graph connectivity") +
    guides(color = FALSE, fill = FALSE, shape = FALSE) +
    ggplot_theme()
ggsave(snakemake@output[["nc_vs_gc_hull"]], gp, width = 3, height = 3)

gp <- ggplot(
    data = df_summarise,
    mapping = aes(x = bio, y = int, color = method)
) +
    geom_errorbar(mapping = aes(
        ymin = int - int_sd,
        ymax = int + int_sd
    ), width = 0) +
    geom_errorbarh(mapping = aes(
        xmin = bio - bio_sd,
        xmax = bio + bio_sd
    ), height = 0) +
    geom_point(data = df, mapping = aes(alpha = alpha)) +
    facet_wrap(~dataset) +
    scale_alpha_continuous(limits = c(0, 1), range = c(0, 1)) +
    scale_color_manual(name = "Method", values = unlist(display[["palette"]])) +
    scale_x_continuous(name = "Biology conservation") +
    scale_y_continuous(name = "Omics mixing") +
    guides(color = FALSE, alpha = FALSE) +
    ggplot_theme()
ggplot_save(snakemake@output[["bio_vs_int"]], gp, width = 5.5, height = 4)

hull <- ddply(
    df_summarise, "method",
    function(df) df %>%
        filter(!is.na(bio) & !is.na(int)) %>%
        slice(chull(bio, int))
) %>% arrange(method)
gp <- ggplot(mapping = aes(
    x = bio, y = int,
    color = method, fill = method
)) +
    geom_polygon(data = hull, alpha = 0.1, linetype="dashed") +
    geom_point(data = df_summarise, mapping = aes(shape = dataset), size = 1.5) +
    scale_color_manual(name = "Method", values = unlist(display[["palette"]])) +
    scale_fill_manual(name = "Method", values = unlist(display[["palette"]])) +
    scale_shape_manual(name = "Dataset", values = unlist(display[["shape"]])) +
    scale_x_continuous(name = "Biology conservation") +
    scale_y_continuous(name = "Omics mixing") +
    guides(color = FALSE, fill = FALSE, shape = FALSE) +
    ggplot_theme()
ggsave(snakemake@output[["bio_vs_int_hull"]], gp, width = 3, height = 3)

gp <- ggplot(
    data = df_summarise,
    mapping = aes(x = dataset, y = overall, fill = method)
) +
    geom_bar(stat = "identity", width = 0.8, position = position_dodge2(width = 0.8)) +
    geom_point(
        data = df, size = 0.2,
        position = position_jitterdodge(jitter.width = 0.3, dodge.width = 0.8),
        show.legend = FALSE
    ) +
    geom_errorbar(
        mapping = aes(ymin = overall - overall_sd, ymax = overall + overall_sd),
        width = 0.2, position = position_dodge(width = 0.8)
    ) +
    geom_text(
        mapping = aes(label = annotation), y = 0.035, vjust = 0.5, size = 2.3,
        angle = 90, position = position_dodge2(width = 0.8), fontface = "bold"
    ) +
    scale_fill_manual(name = "Method", values = unlist(display[["palette"]])) +
    scale_x_discrete(name = "Dataset") +
    scale_y_continuous(name = "Overall integration score") +
    guides(fill = FALSE) +
    ggplot_theme()
ggplot_save(snakemake@output[["overall"]], gp, width = 5, height = 3)

gp <- ggplot(
    data = df_summarise,
    mapping = aes(x = method, y = overall, fill = method)
) +
    geom_bar(stat = "identity") +
    scale_fill_manual(name = "Method", values = unlist(display[["palette"]])) +
    guides(fill = guide_legend(nrow = 3)) +
    ggplot_theme(legend.direction = "horizontal")
leg <- as_ggplot(get_legend(gp))
ggsave(snakemake@output[["legend"]], leg, width = 7, height = 1)

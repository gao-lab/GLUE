suppressPackageStartupMessages({
    source(".Rprofile")
    library(dplyr)
    library(ggplot2)
    library(yaml)
})
set.seed(0)


#-------------------------------- Preparations ---------------------------------

display <- read_yaml("config/display.yaml")

df <- read.csv(snakemake@input[[1]])[, c(
    "dataset", "method", "foscttm",
    "mean_average_precision", "seurat_alignment_score"
)] %>%
    mutate(
        dataset = factor(dataset, levels = names(display[["dataset"]]), labels = display[["dataset"]]),
        method = factor(method, levels = names(display[["method"]]), labels = display[["method"]])
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
        seurat_alignment_score = mean(seurat_alignment_score)
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
            foscttm = NA,
            annotation = "N.A."
        )
        df <- bind_rows(df, placeholder)
        df_summarise <- bind_rows(df_summarise, placeholder)
    }
}


#---------------------------------- Plotting -----------------------------------

gp <- ggplot(data = df_summarise, mapping = aes(x = dataset, y = foscttm, fill = method)) +
    geom_bar(stat = "identity", width = 0.7, position = position_dodge2(width = 0.7)) +
    geom_errorbar(
        mapping = aes(ymin = foscttm - foscttm_sd, ymax = foscttm + foscttm_sd),
        width = 0.2, position = position_dodge(width = 0.7)
    ) +
    geom_text(
        mapping = aes(label = annotation), y = 0.03, vjust = 0.5, size = 2.3,
        angle = 90, position = position_dodge2(width = 0.7), fontface = "bold"
    ) +
    geom_point(
        data = df, size = 0.2,
        position = position_jitterdodge(jitter.width = 0.3, dodge.width = 0.7),
        show.legend = FALSE
    ) +
    scale_fill_manual(name = "Method", values = unlist(display[["palette"]])) +
    scale_x_discrete(name = "Dataset") +
    scale_y_continuous(name = "FOSCTTM") +
    ggplot_theme()
ggplot_save(snakemake@output[["foscttm"]], gp, width = 5.5, height = 2.5)

gp <- ggplot(
    data = df_summarise %>% arrange(desc(method)), mapping = aes(
        x = mean_average_precision,
        y = seurat_alignment_score,
        color = method
    )
) +
    geom_errorbar(
        mapping = aes(
            ymin = seurat_alignment_score - seurat_alignment_score_sd,
            ymax = seurat_alignment_score + seurat_alignment_score_sd
        ), width = 0
    ) +
    geom_errorbarh(
        mapping = aes(
            xmin = mean_average_precision - mean_average_precision_sd,
            xmax = mean_average_precision + mean_average_precision_sd
        ), height = 0
    ) +
    geom_point(data = df %>% arrange(desc(method)), alpha = 0.6) +
    facet_wrap(~dataset) +
    scale_color_manual(name = "Method", values = unlist(display[["palette"]])) +
    scale_x_continuous(name = "Mean average precision") +
    scale_y_continuous(name = "Seurat alignment score") +
    guides(color = FALSE) +
    ggplot_theme()
ggplot_save(snakemake@output[["map_vs_sas"]], gp, width = 5.5, height = 2.5)

suppressPackageStartupMessages({
    source(".Rprofile")
    library(dplyr)
    library(ggplot2)
    library(viridis)
    library(yaml)
})
set.seed(0)


#-------------------------------- Preparations ---------------------------------

display <- read_yaml("config/display.yaml")

df <- read.csv(snakemake@input[[1]])[, c("dataset", "subsample_size", "method", "foscttm")] %>%
    mutate(
        dataset = factor(dataset, levels = names(display[["dataset"]]), labels = display[["dataset"]]),
        method = factor(method, levels = names(display[["method"]]), labels = display[["method"]]),
        subsample_size = factor(subsample_size)
    ) %>%
    as.data.frame()

df_summarise <- df %>%
    group_by(dataset, method, subsample_size) %>%
    summarise(
        foscttm_sd = safe_sd(foscttm),
        foscttm = mean(foscttm)
    ) %>%
    as.data.frame()


#---------------------------------- Plotting -----------------------------------

gp <- ggplot(data = df_summarise, mapping = aes(
    x = subsample_size, y = foscttm,
    ymin = foscttm - foscttm_sd, ymax = foscttm + foscttm_sd,
    group = method, color = method
)) +
    geom_point() + geom_line() + geom_errorbar(width = 0.1) +
    facet_wrap(~ dataset) +
    scale_color_manual(name = "Method", values = unlist(display[["palette"]])) +
    scale_x_discrete(name = "Subsample size") +
    scale_y_continuous(name = "FOSCTTM") +
    guides(color = FALSE) +
    ggplot_theme(axis.text.x = element_text(angle = 60, vjust = 0.5))
ggplot_save(snakemake@output[[1]], gp, width = 5.5, height = 2.5)

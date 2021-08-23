suppressPackageStartupMessages({
    source(".Rprofile")
    library(dplyr)
    library(ggplot2)
    library(rlang)
    library(yaml)
})
set.seed(0)


#-------------------------------- Preparations ---------------------------------

display <- read_yaml("config/display.yaml")
params <- snakemake@config$method$GLUE
defaults <- lapply(params, function(x) x$default)

df <- read.csv(snakemake@input[[1]])[, c("dataset", names(params), "foscttm")]

df_list <- list()
for (param in names(params)) {
    df_filter <- duplicate(df)
    for (constant in names(params)) {
        if (param == constant) next
        df_filter <- df_filter %>%
            filter(!!as.symbol(constant) == defaults[[constant]])
    }
    df_filter <- df_filter %>%
        select(dataset, !!as.symbol(param), foscttm) %>%
        rename(value = !!as.symbol(param))
    df_list[[param]] <- df_filter
}

df <- bind_rows(df_list, .id = "param") %>%
    mutate(
        dataset = factor(dataset, levels = names(display[["dataset"]]), labels = display[["dataset"]]),
        param = factor(param, levels = names(display[["param"]]), labels = display[["param"]]),
        value = factor(as.character(value), as.character(sort(unique(value))))
    ) %>%
    group_by(dataset, param, value) %>%
    summarise(foscttm_sd = sd(foscttm), foscttm = mean(foscttm)) %>%
    as.data.frame()


#---------------------------------- Plotting -----------------------------------

gp <- ggplot(data = df, mapping = aes(
    x = value, y = foscttm,
    ymax = foscttm + foscttm_sd,
    ymin = foscttm - foscttm_sd,
    color = dataset, group = dataset
)) +
    geom_point() + geom_line() + geom_errorbar(width = 0.1) +
    facet_wrap(~ param, scales = "free_x") +
    scale_x_discrete(name = "Hyperparameter value") +
    scale_y_continuous(name = "FOSCTTM") +
    scale_color_discrete(name = "Dataset") +
    ggplot_theme()
ggplot_save(snakemake@output[[1]], gp, width = 8, height = 7)

suppressPackageStartupMessages({
    source(".Rprofile")
    library(dplyr)
    library(ggplot2)
    library(yaml)
})
set.seed(0)


#-------------------------------- Preparations ---------------------------------

display <- read_yaml("display.yaml")

df <- read.csv(snakemake@input[[1]])[, c("method", "n_cells", "time")] %>%
    mutate(
        method = factor(method, levels = names(display[["method"]]), labels = display[["method"]])
    ) %>%
    as.data.frame()

df_summarise <- df %>%
    group_by(method) %>%
    mutate(slope = lm(y ~ x, data.frame(x = log10(n_cells), y = log10(time)))$coefficients["x"]) %>%
    mutate(slope = sprintf("%.3f", slope)) %>%
    as.data.frame()

ms <- as.list(df_summarise$slope)
names(ms) <- df_summarise$method
ms <- ms[!duplicated(names(ms))]
for (m in names(ms)) {
    ms[[m]] <- bquote(.(m) ~ "(" * beta == .(ms[[m]]) * ")")
}  # Method-slope mapping for labels


#---------------------------------- Plotting -----------------------------------

gp <- ggplot(data = df_summarise, mapping = aes(x = n_cells, y = time, col = method)) +
    geom_line() + geom_point() +
    scale_color_manual(name = "Method", values = unlist(display[["palette"]]), labels = ms) +
    scale_x_log10(name = "Number of cells") +
    scale_y_log10(name = "Time (s)") +
    ggplot_theme()
ggplot_save(snakemake@output[[1]], gp, width = 4.5, height = 2.5)

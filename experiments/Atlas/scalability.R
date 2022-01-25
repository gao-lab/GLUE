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
    mutate(slope = lm(y ~ x, data.frame(x = log10(n_cells), y = log10(time)))$coefficients["x"])
df_slope <- df_summarise %>% select(method, slope) %>% distinct() %>% as.data.frame()

ms <- as.list(df_slope %>% mutate(slope = sprintf("%.3f", slope)) %>% pull(slope))
names(ms) <- df_slope$method
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


gp <- ggplot(data = df_slope, mapping = aes(x = method, y = slope, fill = method)) +
    geom_bar(stat = "identity") +
    geom_hline(yintercept = 1, linetype = "dashed", color = "grey") +
    scale_x_discrete(name = "Method") +
    scale_y_continuous(name = expression("Slope (" * beta * ")")) +
    scale_fill_manual(name = "Method", values = unlist(display[["palette"]])) +
    ggplot_theme(axis.text.x = element_text(angle = 60, vjust = 1.0, hjust = 1.0))
ggplot_save(snakemake@output[[2]], gp, width = 3, height = 2.5)

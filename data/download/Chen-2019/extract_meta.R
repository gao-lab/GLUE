#!/usr/bin/env Rscript

meta <- readRDS("AdBrainCortex_SNAREseq_metadata.rds")
write.csv(meta, "AdBrainCortex_SNAREseq_metadata.csv", quote=FALSE)

suppressPackageStartupMessages({
    source("../../../.Rprofile", chdir = TRUE)
    library(Signac)
    library(Seurat)
    library(GenomeInfoDb)
    library(EnsDb.Mmusculus.v79)
    library(ggplot2)
    library(patchwork)
})
set.seed(1234)

counts <- Read10X_h5('atac_v1_adult_brain_fresh_5k_filtered_peak_bc_matrix.h5')
metadata <- read.csv(
  file = 'atac_v1_adult_brain_fresh_5k_singlecell.csv',
  header = TRUE,
  row.names = 1
)

brain_assay <- CreateChromatinAssay(
  counts = counts,
  sep = c(":", "-"),
  genome = "mm10",
  fragments = 'atac_v1_adult_brain_fresh_5k_fragments.tsv.gz',
  min.cells = 1
)

brain <- CreateSeuratObject(
  counts = brain_assay,
  assay = 'peaks',
  project = 'ATAC',
  meta.data = metadata
)

# extract gene annotations from EnsDb
annotations <- GetGRangesFromEnsDb(ensdb = EnsDb.Mmusculus.v79)

# change to UCSC style since the data was mapped to hg19
seqlevelsStyle(annotations) <- 'UCSC'
genome(annotations) <- "mm10"

# add the gene information to the object
Annotation(brain) <- annotations

brain <- NucleosomeSignal(object = brain)

brain$nucleosome_group <- ifelse(brain$nucleosome_signal > 4, 'NS > 4', 'NS < 4')
FragmentHistogram(object = brain, group.by = 'nucleosome_group', region = 'chr1-1-10000000')

brain <- TSSEnrichment(brain, fast = FALSE)

brain$high.tss <- ifelse(brain$TSS.enrichment > 2, 'High', 'Low')
TSSPlot(brain, group.by = 'high.tss') + NoLegend()

brain$pct_reads_in_peaks <- brain$peak_region_fragments / brain$passed_filters * 100
brain$blacklist_ratio <- brain$blacklist_region_fragments / brain$peak_region_fragments

VlnPlot(
  object = brain,
  features = c('pct_reads_in_peaks', 'peak_region_fragments',
               'TSS.enrichment', 'blacklist_ratio', 'nucleosome_signal'),
  pt.size = 0.1,
  ncol = 5
)

brain <- subset(
  x = brain,
  subset = peak_region_fragments > 3000 &
    peak_region_fragments < 100000 &
    pct_reads_in_peaks > 40 &
    blacklist_ratio < 0.025 &
    nucleosome_signal < 4 &
    TSS.enrichment > 2
)
brain

brain <- RunTFIDF(brain)
brain <- FindTopFeatures(brain, min.cutoff = 'q0')
brain <- RunSVD(object = brain)

DepthCor(brain)

brain <- RunUMAP(
  object = brain,
  reduction = 'lsi',
  dims = 2:30
)
brain <- FindNeighbors(
  object = brain,
  reduction = 'lsi',
  dims = 2:30
)
brain <- FindClusters(
  object = brain,
  algorithm = 3,
  resolution = 1.2,
  verbose = FALSE
)

DimPlot(object = brain, label = TRUE) + NoLegend()

# compute gene activities
gene.activities <- GeneActivity(brain)

# add the gene activity matrix to the Seurat object as a new assay
brain[['RNA']] <- CreateAssayObject(counts = gene.activities)
brain <- NormalizeData(
  object = brain,
  assay = 'RNA',
  normalization.method = 'LogNormalize',
  scale.factor = median(brain$nCount_RNA)
)

DefaultAssay(brain) <- 'RNA'
FeaturePlot(
  object = brain,
  features = c('Sst','Pvalb',"Gad2","Neurod6","Rorb","Syt6"),
  pt.size = 0.1,
  max.cutoff = 'q95',
  ncol = 3
)

# Load the pre-processed scRNA-seq data
allen_rna <- readRDS("allen_brain.rds")
allen_rna <- FindVariableFeatures(
  object = allen_rna,
  nfeatures = 5000
)

transfer.anchors <- FindTransferAnchors(
  reference = allen_rna,
  query = brain,
  reduction = 'cca',
  dims = 1:40
)

predicted.labels <- TransferData(
  anchorset = transfer.anchors,
  refdata = allen_rna$subclass,
  weight.reduction = brain[['lsi']],
  dims = 2:30
)

brain <- AddMetaData(object = brain, metadata = predicted.labels)

plot1 <- DimPlot(allen_rna, group.by = 'subclass', label = TRUE, repel = TRUE) + NoLegend() + ggtitle('scRNA-seq')
plot2 <- DimPlot(brain, group.by = 'predicted.id', label = TRUE, repel = TRUE) + NoLegend() + ggtitle('scATAC-seq')
plot1 + plot2

# replace each label with its most likely prediction
for(i in levels(brain)) {
  cells_to_reid <- WhichCells(brain, idents = i)
  newid <- names(sort(table(brain$predicted.id[cells_to_reid]),decreasing=TRUE))[1]
  Idents(brain, cells = cells_to_reid) <- newid
}

write.csv(brain@meta.data, "signac_meta_data.csv", quote=FALSE)

brain.idents <- as.data.frame(Idents(brain))
colnames(brain.idents) <- "Idents"
write.csv(brain.idents, "signac_idents.csv", quote=FALSE)

DimPlot(brain, label = TRUE, repel = TRUE)

#switch back to working with peaks instead of gene activities
DefaultAssay(brain) <- 'peaks'

da_peaks <- FindMarkers(
  object = brain,
  ident.1 = c("L2/3 IT"),
  ident.2 = c("L4", "L5 IT", "L6 IT"),
  min.pct = 0.4,
  test.use = 'LR',
  latent.vars = 'peak_region_fragments'
)

head(da_peaks)

plot1 <- VlnPlot(
  object = brain,
  features = rownames(da_peaks)[1],
  pt.size = 0.1,
  idents = c("L4","L5 IT","L2/3 IT")
)
plot2 <- FeaturePlot(
  object = brain,
  features = rownames(da_peaks)[1],
  pt.size = 0.1,
  max.cutoff = 'q95'
)
plot1 | plot2

open_l23 <- rownames(da_peaks[da_peaks$avg_logFC > 0.25, ])
open_l456 <- rownames(da_peaks[da_peaks$avg_logFC < -0.25, ])
closest_l23 <- ClosestFeature(brain, open_l23)
closest_l456 <- ClosestFeature(brain, open_l456)
head(closest_l23)

head(closest_l456)

# set plotting order
levels(brain) <- c("L2/3 IT","L4","L5 IT","L5 PT","L6 CT", "L6 IT","NP","Sst","Pvalb","Vip","Lamp5","Meis2","Oligo","Astro","Endo","VLMC","Macrophage")

CoveragePlot(
  object = brain,
  region = c("Neurod6", "Gad2"),
  extend.upstream = 1000,
  extend.downstream = 1000,
  ncol = 1
)

" Goal: Cleanup source data for P7 kdr-ko scRNAseq performed in September 2025
Date:250909
Author:Carsten Knutsen
conda_env:soupxR
"
#install.packages('SoupX')
#install.packages('installr')
#library(installr)
#install.Rtools()
#install.packages('Matrix')
#BiocManager::install("DropletUtils")
library(DropletUtils)
library(Matrix)
library(SoupX)
library(Seurat)
library(ggplot2)
graphics.off() 
par("mar") 
par(mar=c(1,1,1,1))
getwd()
data_dir <- 'data/single_cell_files/cellranger_output'
sub_fols <- list.dirs(path = data_dir, full.names = FALSE, recursive = FALSE)
output_dir <-'data/single_cell_files/soupx'
dir.create(output_dir, recursive = TRUE)
figs_dir <- 'data/figures/soupx'
dir.create(figs_dir, recursive = TRUE)
for (fol in sub_fols)
{
  print(fol)
  subdir <- sprintf('%s/%s/outs',data_dir,fol)
  subdir_out <- sprintf('%s/%s',output_dir,fol)
  if (dir.exists(subdir_out)) {
    print(paste("File", subdir_out, "does not exist. Skipping to next file."))
    next # Skips the rest of the current loop iteration and moves to the next
  }
  cellnames <- read.csv(sprintf('%s/filtered_feature_bc_matrix/barcodes.tsv.gz', subdir),header =FALSE)
  filt.matrix <- Read10X_h5(sprintf("%s/filtered_feature_bc_matrix.h5",subdir),use.names = F)
  raw.matrix <- Read10X_h5(sprintf("%s/raw_feature_bc_matrix.h5",subdir),use.names = F)
  soup.channel = SoupChannel(raw.matrix, filt.matrix)
  srat <- CreateSeuratObject(counts = filt.matrix)
  srat <-RenameCells(srat, new.names = cellnames$V1)
  srat    <- SCTransform(srat, verbose = F)
  srat    <- RunPCA(srat, verbose = F)
  srat    <- RunUMAP(srat, dims = 1:30, verbose = F)
  srat    <- FindNeighbors(srat, dims = 1:30, verbose = F)
  srat    <- FindClusters(srat, verbose = T)
  meta    <- srat@meta.data
  umap    <- srat@reductions$umap@cell.embeddings
  png(sprintf("%s/%s.png",figs_dir,fol),width = 5, height = 4, units = 'in',res=300)
  soup.channel  <- setClusters(soup.channel, setNames(meta$seurat_clusters, rownames(meta)))
  soup.channel  <- setDR(soup.channel, umap)
  soup.channel  <- autoEstCont(soup.channel,forceAccept=TRUE)
  gg <- plotMarkerDistribution(soup.channel)
  ggsave(sprintf("%s/%s_marker_distribution.png",figs_dir,fol),
         gg)
  adj.matrix  <- adjustCounts(soup.channel, roundToInt = T)
  DropletUtils:::write10xCounts(subdir_out, adj.matrix)
  ambient_genes_df <- soup.channel$soupProfile[order(soup.channel$soupProfile$est, decreasing = TRUE), ]
  output_file <-sprintf("%s/%s_ambient_genes.txt",figs_dir,fol)
  write.table(ambient_genes_df, file = output_file, row.names = TRUE, col.names = NA, quote = FALSE, sep = "\t")
  dev.off()
}


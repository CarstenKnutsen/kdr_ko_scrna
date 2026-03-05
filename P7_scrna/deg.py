"""
Goal: Run differential expression on Library for all cell types
Author:Carsten Knutsen
Date:251030
conda_env:vegfr2
"""


import scanpy as sc
import scanpy.external as sce
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread
import matplotlib.pyplot as plt
import os
from functions import compare_obs_values_within_groups_to_excel

adata_name = "kdr_ko_p7_sc"
output = "data/figures/deg"
os.makedirs(output, exist_ok=True)
data = "data/single_cell_files/scanpy_files"
os.makedirs(output, exist_ok=True)
sc.set_figure_params(dpi=300, dpi_save=300, format="png")
sc.settings.figdir = output

if __name__ == "__main__":
    adata = sc.read(f"{data}/{adata_name}_celltyped.gz.h5ad")
    adata.uns['log1p']['base'] = None
    adata = adata[:,(adata.var['mt']==False)&(adata.var['ribo']==False)&(adata.var['hb']==False)]
    compare_obs_values_within_groups_to_excel(adata,'Library',group_column='Cell Subtype',output_prefix=f"{output}/library_degs")
    for ct in adata.obs['Cell Subtype'].cat.categories:
        ct_adata = adata[adata.obs['Cell Subtype']==ct]
        sc.tl.rank_genes_groups(ct_adata, groupby=f"Library", method='wilcoxon', pts=True)
        sc.tl.dendrogram(ct_adata,"Library")
        sc.pl.rank_genes_groups_dotplot(
            ct_adata,
            groupby="Library",
            show=False,
            save=f"{ct}_library_markers.png",
        )
        with pd.ExcelWriter(
                f"{output}/library_markers_{ct}.xlsx", engine="xlsxwriter"
        ) as writer:
            for lib in ct_adata.obs["Library"].cat.categories:
                df = sc.get.rank_genes_groups_df(ct_adata, key="rank_genes_groups", group=lib)
                df.set_index("names")
                df["pct_difference"] = df["pct_nz_group"] - df["pct_nz_reference"]
                df.to_excel(writer, sheet_name=f"{lib} v rest"[:31])
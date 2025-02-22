"""
Goal: Run differential expression on Library for all cell types
Author:Carsten Knutsen
Date:250103
conda_env:kdr_ko
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

adata_name = "kdr_ko_p7"
output = "data/figures/deg_no_cc"
os.makedirs(output, exist_ok=True)
data = "data/single_cell_files/scanpy_files"
os.makedirs(output, exist_ok=True)
sc.set_figure_params(dpi=300, dpi_save=300, format="png")
sc.settings.figdir = output

if __name__ == "__main__":
    adata = sc.read(f"{data}/{adata_name}_celltyped_no_cc.gz.h5ad")
    adata.uns['log1p']['base'] = None
    adata = adata[:,(adata.var['mt']==False)&(adata.var['ribo']==False)&(adata.var['hb']==False)]
    compare_obs_values_within_groups_to_excel(adata,'Library',group_column='Cell Subtype_no_cc',output_prefix=f"{output}/library_degs")
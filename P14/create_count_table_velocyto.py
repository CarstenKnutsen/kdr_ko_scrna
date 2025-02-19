'''Goal:Create count table for all cells after SoupX cleanup of tbx2 ko samples
Date:231222
Author: Carsten Knutsen
conda_env:trajectory_inference
'''


import os
import scanpy as sc
import scvelo as scv
import numpy as np
import pandas as pd
import string
import anndata
from gtfparse import read_gtf
from anndata import AnnData
from collections import defaultdict
data = 'data/single_cell_files/cellranger_output'
output = 'data/single_cell_files/scanpy_files'
adata_name = "kdr_ko_p14"

os.makedirs(output, exist_ok=True)
if __name__ == '__main__':
    runs = os.listdir(data)
    adatas = []
    for run in runs:
        print(run)
        fn = f'{data}/{run}/velocyto/{run}.loom'
        adata_v = sc.read(fn)
        adata_v.var_names_make_unique()
        adata_v.obs_names = adata_v.obs_names.str.replace(':', '_')  # to match original data
        adata_v.obs_names = adata_v.obs_names.str.rstrip('x')
        adata_v.obs_names = [x + '-1' for x in adata_v.obs_names]

        meta = run.split("_")[0]
        adata_v.obs["Treatment"] = meta[0]
        adata_v.obs["Timepoint"] = meta[-3:]
        adata_v.obs["Genotype"] = meta[-5:-3]
        adata_v.obs["Library"] = adata_v.obs["Treatment"] + '_' + adata_v.obs["Genotype"]
        adatas.append(adata_v.copy())
    adata_v = anndata.concat(adatas)
    adata_v.obs["Treatment"].replace({"H": "Hyperoxia", "N": "Normoxia"}, inplace=True)
    print(adata_v)
    adata_v.write(f'{output}/{adata_name}_all_cells_velocyto.gz.h5ad', compression='gzip')

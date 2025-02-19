"""
Goal: clean up h5ad for sharing
Author:Carsten Knutsen
Date:250203
conda_env:kdr_ko
"""


import scanpy as sc
import pandas as pd
import os


adata_name = "kdr_ko_p14"
data = "data/single_cell_files/scanpy_files"
os.makedirs(data, exist_ok=True)

if __name__ == "__main__":
    adata = sc.read(f"{data}/{adata_name}_celltyped_no_cc.gz.h5ad")
    adata.X = adata.layers['log1p'].copy()
    adata.uns['log1p']['base'] = None
    del adata.layers['log1p_cc_regress']
    adata.write(f'{data}/{adata_name}_share.gz.h5ad',compression='gzip')
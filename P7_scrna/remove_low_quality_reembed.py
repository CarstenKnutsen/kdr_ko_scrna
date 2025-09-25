"""Goal:Initial analysis of P7 Kdr KO scRNAseq, creating h5ad, filtering, and embedding
Date:250203
Author: Carsten Knutsen
conda_env:vegfr2
"""

import pandas as pd
import os
import scanpy as sc
import seaborn as sns
import matplotlib.pylab as plt
import numpy as np
from scipy.stats import median_abs_deviation
import scanpy.external as sce
import itertools
from gtfparse import read_gtf
import anndata
from collections import defaultdict

#we set hardcoded paths here
gtf_fn = 'data/genes.gtf' ## Downloaded from 10X provided refdata-gex-GRCm39-2024-A
# input_data = "data/single_cell_files/cellranger_output"
input_data = "data/single_cell_files/soupx"

output_data = "data/single_cell_files/scanpy_files"
os.makedirs(output_data, exist_ok=True)
adata_name = "kdr_ko_p7_sc"
figures = "data/figures"
os.makedirs(figures, exist_ok=True)
sc.set_figure_params(dpi=300, format="png")
sc.settings.figdir = figures
#usefula gene lists below
gene_dict = {
        "mesenchymal": [
            "Zbtb16",
            "Col3a1",
            "G0s2",
            "Limch1",
            "Col13a1",
            "Col14a1",
            "Serpinf1",
            "Pdgfra",
            "Scara5",
            "Acta2",
            "Hhip",
            "Fgf18",
            "Wif1",
            "Tgfbi",
            "Tagln",
            "Mustn1",
            "Aard",
            "Pdgfrb",
            "Notch3",
            "Dcn",
            "Cox4i2",
            "Higd1b",
            "Wt1",
            "Lrrn4",
            "Upk3b",
            "Mki67",
            'Lgals3',
            # 'Tubb3',
            # 'Aqp3',
'Ttn','Ryr2','Myh6','Tbx20','Ldb3','Eya4','Rbm20','Neb','Itm2a',
            'Col1a1',
            "Epcam",
            "Ptprc",
            "Pecam1",
            'Zbtb16',
        ],
        "endothelial": [
"Zbtb16",
            "Gja5",
            "Dkk2",
            "Bmx",
            "Fn1",
            "Ctsh",
            "Kcne3",
            "Cdh13",
            "Car8",
            "Mmp16",
            "Slc6a2",
            "Thy1",
            "Mmrn1",
            "Ccl21a",
            "Reln",
            "Neil3",
            "Mki67",
            "Aurkb",
            "Depp1",
            "Ankrd37",
            "Peg3",
            "Mest",
            "Hpgd",
            "Cd36",
            "Car4",
            'Apln',
            'Aplnr',
            "Sirpa",
            "Fibin",
            'Tbx2',
            'Kit',
            "Col15a1",
            "Col1a1",
            "Epcam",
            "Ptprc",
            "Pecam1",
            'Zbtb16',
        ],
        "immune": [
"Zbtb16",
            "Cd68",
            # "Gal",
            "Itgax",
            "Car4",
            "C1qa",
            "Plac8",
            "Batf3",
            "Itgae",
            # "Cd209a",
            "Mreg",
            # "Mcpt4",
            # "Mcpt8",
            "Retnlg",
            "Ms4a1",
            "Gzma",
            "Cd3e",
            "Areg",
            "Mki67",
            "Col1a1",
            "Epcam",
            "Ptprc",
            "Pecam1",
        ],
        "epithelial": [
"Zbtb16",
            'Muc1',
            "Scg5",
            "Ascl1",
            "Lyz1",
            "Lyz2",
            "Sftpc",
            "Slc34a2",
            "S100g",
            "Sftpa1",
            "Akap5",
            "Hopx",
            "Col4a4",
            "Vegfa",
            "Lyz1",
            "Tmem212",
            "Dynlrb2",
            "Cdkn1c",
            "Tppp3",
            "Scgb3a2",
            "Cyp2f2",
            "Scgb1a1",
            "Reg3g",
            "Scgb3a1",
            "Mki67",
            "Col1a1",
            "Epcam",
            "Ptprc",
            "Pecam1",
            'Dnah12',
            'Spag17',
            'Muc5b',

        ],
    }#leiden dictionary to assign cell types
leiden_ct_dict = {
    "0": "Cap1",
    "1": "Cap2",
    "2": "Alveolar fibroblast",
    "3": "Myofibroblast",
    "4": "low-quality-Alveolar fibroblast",
    "5": "low-quality cap1",
    "6": "AT1",
    "7": "Adventitial fibroblast",
    "8": "AT2",
    "9": "Myofibroblast",
    "10":  "Vascular smooth muscle",
    "11":"Arterial EC",
    "12": "Pericyte",
    "13": "low-quality zbtb16 cap1",
    "14": "low-quality zbtb16 alv fib",
    "15":"low-quality zbtb16 myofib",
    "16": "AT2",

    "17": 'doublet EC',
        "18": "Proliferating EC",
    "19": "Venous EC",
"20": "Mesothelial",
"21": "low-quality zbtb16 cap2",
"22": "low-quality doublet",
"23": "Lymphatic EC",
    "24": "Ciliated",
    "25": "Proliferating mese",
    "26": "Basophil",
    "27": "Cardiomyocyte",
}
if __name__ == "__main__":
    adata = sc.read(f"{output_data}/{adata_name}_filtered_embed_w_doublets.gz.h5ad",)
    print(adata)
    adata = adata[
        (~adata.obs.predicted_doublet)
        # &(adata.obs.doublet_score<0.1)
                  ].copy()
    print(adata)
    # # run embedding and clustering below
    figures_embed = f"{figures}/initial_embedding_remove_doublets_clean"
    os.makedirs(figures_embed, exist_ok=True)
    sc.settings.figdir = figures_embed
    sc.pp.highly_variable_genes(adata, batch_key="Library")
    sc.pp.pca(adata, mask_var="highly_variable")
    # sce.pp.harmony_integrate(adata, 'Library',adjusted_basis='X_pca')
    sc.pp.neighbors(adata, use_rep="X_pca")
    sc.tl.leiden(adata, key_added="leiden", resolution=0.5)
    sc.tl.umap(adata, min_dist=0.5)
    sc.tl.score_genes(adata, adata.var['ambient_rna_est_contamination'].sort_values(ascending=False).head(50).index,
                      score_name='ambient_score')

    genes = ['Col1a1', 'Cdh5', 'Ptprc', 'Cdh1', 'leiden']
    genedf = sc.get.obs_df(adata, keys=genes)
    grouped = genedf.groupby("leiden")
    mean = grouped.mean()
    mean_t = mean.T
    lineage_dict = {}
    for cluster in mean_t.columns:
        gene = mean_t[cluster].idxmax()
        if gene == 'Cdh5':
            lineage_dict[cluster] = 'Endothelial'
        elif gene == 'Ptprc':
            lineage_dict[cluster] = 'Immune'
        elif gene == 'Cdh1':
            lineage_dict[cluster] = 'Epithelial'
        elif gene == 'Col1a1':
            lineage_dict[cluster] = 'Mesenchymal'
    mean_t.to_csv(f'{figures_embed}/lineage_scores.csv')
    adata.uns['leiden_lineage_expression'] = mean_t
    adata.obs['Lineage'] = [lineage_dict[x] if x != '27' else 'Mesenchymal' for x in adata.obs['leiden']]
    adata.obs["celltype_rough"] = [leiden_ct_dict[x] for x in adata.obs["leiden"]]

    ## make plots below that are helpful for initial analysis
    sc.pl.dotplot(
        adata,
        ['Cdh5', 'Col1a1', 'Ptprc', 'Epcam', 'Muc1', 'Muc5b', 'Foxa2', 'Cdh1', 'Krt8', 'Sox9', 'Tbx2', 'Kit', 'Car4',
         'Sftpc', 'Hopx', 'Zbtb16', 'Ccnd3', 'Fkbp5'],
        groupby="leiden",
        dendrogram=False,
        show=False,
        save="useful_genes.png",
    )
    for lin, genes in gene_dict.items():
        sc.pl.dotplot(
            adata,
            [x for x in genes if x in adata.var_names],
            groupby="leiden",
            dendrogram=False,
            show=False,
            save=f"useful_genes_{lin}.png",
        )
    for color in [
        "log1p_total_umis",
        "log1p_n_genes_by_umis",
        "Library",
        "Treatment",
        "Genotype",
        "Lineage",
        "leiden",
        "doublet_score",
        "predicted_doublet",
        "celltype_rough",
        "ambient_score",
        'Cdh5', 'Col1a1', 'Ptprc', 'Epcam', 'Muc1', 'Tbx2', 'Kit', 'Car4', 'Sftpc', 'Hopx', 'Zbtb16'
    ]:
        sc.pl.umap(adata, color=color, show=False, save=f"_{color}.png")
    adata =adata[(~adata.obs['celltype_rough'].str.startswith('doublet'))&(~adata.obs['celltype_rough'].str.startswith('low-quality'))]
    sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon", pts=True)
    with pd.ExcelWriter(
            f"{figures_embed}/leiden_markers.xlsx", engine="xlsxwriter"
    ) as writer:
        for ct in adata.obs["leiden"].cat.categories:
            df = sc.get.rank_genes_groups_df(adata, key="rank_genes_groups", group=ct)
            df.set_index("names")
            df["pct_difference"] = df["pct_nz_group"] - df["pct_nz_reference"]
            df.to_excel(writer, sheet_name=f"{ct} v rest"[:31])
    sc.pl.rank_genes_groups_dotplot(
        adata,
        groupby="leiden",
        n_genes=int(150 / len(adata.obs["leiden"].unique())),
        show=False,
        save=f"leiden_markers.png",
    )
    sc.pl.pca_overview(adata, color="leiden", show=False, save=True)
    sc.pl.pca_variance_ratio(adata, show=False, save=True)
    sc.pl.pca_loadings(
        adata,
        components=",".join([str(x) for x in range(1, 10)]),
        show=False,
        save=True,
    )
    with pd.ExcelWriter(
            f"{figures_embed}/metadata_counts.xlsx", engine="xlsxwriter"
    ) as writer:
        obs_list = ["Library", "Treatment", "leiden"]
        num_obs = len(obs_list) + 1
        for ind in range(0, num_obs):
            for subset in itertools.combinations(obs_list, ind):
                if len(subset) != 0:
                    subset = list(subset)
                    if len(subset) == 1:
                        key = subset[0]
                        adata.obs[key].value_counts().to_excel(writer, sheet_name=key)
                    else:
                        key = "_".join(subset)
                        adata.obs.groupby(subset[:-1])[subset[-1]].value_counts(
                            normalize=True
                        ).to_excel(writer, sheet_name=key[:31])
    for lineage in adata.obs['Lineage'].cat.categories:
        lin_adata = adata[adata.obs['Lineage'] == lineage].copy()
        sc.pl.umap(lin_adata, color=['leiden',
                                     'celltype_rough',
                                     'Library'
                                     ],
                   ncols=1, save=f'{lineage}', show=False)
        print(lin_adata)
        if lineage == 'Immune':
            continue
        else:
            sc.tl.dendrogram(lin_adata, groupby='leiden')
            sc.tl.rank_genes_groups(lin_adata, groupby='leiden')
            sc.pl.rank_genes_groups_dotplot(lin_adata, save=f'{lineage}', show=False)
    adata.write(
        f"{output_data}/{adata_name}_filtered_embed_clean.gz.h5ad", compression="gzip"
    )


"""Goal:Cell typing by lineage
Date:250203
Author: Carsten Knutsen
conda_env:kdr_ko
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
import string
from gtfparse import read_gtf
import anndata
from collections import defaultdict
from functions import plot_obs_abundance
#we set hardcoded paths here
data = "data/single_cell_files/scanpy_files"
adata_name = "kdr_ko_p14"
figures = "data/figures/cell_typing"
os.makedirs(figures, exist_ok=True)
sc.settings.figdir = figures
sc.set_figure_params(dpi=300, format="png")

gene_dict = {
        "mesenchymal": [
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
            "Gja5",
            "Dkk2",
            'Apln',
            'Aplnr',
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
            "Sirpa",
            "Fibin",
            'Tbx2',
            "Col15a1",
            "Col1a1",
            "Epcam",
            "Ptprc",
            "Pecam1",
        ],
        "immune": [
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
            "Mcpt4",
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
    }
leiden_ct_dict = {
    "Mesenchymal": {
        "0": "Alveolar fibroblast",
        "1": "Alveolar fibroblast",
        "2": "Airway smooth muscle_Myofibroblast",
        "3": "Mesothelial",
        "4": "Mural cell",
        "5": "Mural cell",
        "6": "Airway smooth muscle_Myofibroblast",
        "7": "Alveolar fibroblast",
        "8": "Adventitial fibroblast",
        "9":"doublet-fibroblast",
        "10":  "doublet-fibroblast",

    },
    "Endothelial": {
        "0": "Cap1",
        "1": "Cap2",
        "2": "Cap1",
        "3": "Cap2",
        "4": "Cap1_Cap2",
        "5": "Venous EC",
        "6": "Cap1_Cap2",
        "7": "Arterial EC",
        "8": "Proliferating EC",
        "9":  "Lymphatic EC",
    },
    "Immune": {
        "0": "Alveolar macrophage",
        "1": "Alveolar macrophage",
        "2": "low-quality-Alveolar macrophage",
        "3": "T cell",
        "4": "B cell",
        "5": "Proliferating Alveolar macrophage",
        "6": "Interstitial macrophage",
        "7": "Monocyte",
        "8": "Dendritic cell",
        "9":"Dendritic cell",
        "10":  "doublet-",
        "11": "Interstitial macrophage",
        "12": "B cell",
        "13": "Alveolar macrophage",
        "14": "Neutrophil",
        "15": "Proliferating B cell",
        "16": "c-Dendritic cell",
        "17": "mig-Dendritic cell",
    },
    "Epithelial": {
        "0": "AT1",
        "1": "AT2",
        "2": "Club",
        "3": "AT2",
        "4": "AT1",
        "5": "Ciliated",
        "6": "AT1_AT2",
        "7": "AT2",
        "8": "AT1_AT2",
        "9": "Club",
        "10": "Neuroendocrine",

    },

}
if __name__ == "__main__":
    adata = sc.read(
        f"{data}/{adata_name}_filtered_embed.gz.h5ad",
    )
    print(adata)
    adata.obs["Cell Subtype"] = pd.Series(index=adata.obs.index, data=None, dtype="str")
    for lineage in adata.obs['Lineage'].cat.categories:
        figures_lin = f"data/figures/cell_typing/{lineage}"
        os.makedirs(figures_lin, exist_ok=True)
        sc.settings.figdir = figures_lin
        print(lineage)
        lin_adata = adata[adata.obs['Lineage'] == lineage]
        sc.pp.highly_variable_genes(lin_adata, batch_key="Library")
        sc.pp.pca(lin_adata, mask_var="highly_variable")
        sc.pp.neighbors(lin_adata, use_rep="X_pca")
        sc.tl.leiden(lin_adata, key_added=f"leiden_{lineage}", resolution=0.5)
        sc.tl.umap(lin_adata,min_dist=0.5)
        sc.tl.rank_genes_groups(lin_adata, groupby=f"leiden_{lineage}",method='wilcoxon',pts=True)
        sc.tl.dendrogram(lin_adata,f"leiden_{lineage}")
        sc.pl.rank_genes_groups_dotplot(
            lin_adata,
            groupby=f"leiden_{lineage}",
            show=False,
            save=f"leiden_markers.png",
        )
        with pd.ExcelWriter(
                f"{figures_lin}/{lineage}_leiden_markers.xlsx", engine="xlsxwriter"
        ) as writer:
            for ct in lin_adata.obs[f"leiden_{lineage}"].cat.categories:
                df = sc.get.rank_genes_groups_df(lin_adata, key="rank_genes_groups", group=ct)
                df.set_index("names")
                df["pct_difference"] = df["pct_nz_group"] - df["pct_nz_reference"]
                df.to_excel(writer, sheet_name=f"{ct} v rest"[:31])
        sc.pl.umap(lin_adata, color = ['leiden',f"leiden_{lineage}",'celltype_rough'],wspace=0.5,show=False,save='_pretrim_leiden')
        sc.pl.dotplot(lin_adata,gene_dict[lineage.lower()],groupby=f"leiden_{lineage}",show=False,save='useful_genes_leiden')
        sc.pl.umap(lin_adata, color = gene_dict[lineage.lower()],wspace=0.5,show=False,save='_pretrim_genes')
        sc.pl.umap(lin_adata, color = ['log1p_total_umis','log1p_n_genes_by_umis','doublet_score','ambient_score'],wspace=0.5,show=False,save='_pretrim_qc')
        if lineage == 'Mesenchymal':
            weird_cells = {'striated_muscle': ['Ttn', 'Ryr2', 'Myh6', 'Tbx20', 'Ldb3'],
                         'multi_ab_musc': ['Gm20754', 'Pdzrn4', 'Chrm2', 'Cacna2d3', 'Chrm3'],
                         'multi_acta1': ['Eya4', 'Rbm20', 'Neb', 'Itm2a'],
                         'male hyperoxic fibroblast': [ 'Actc1', 'Tuba4a'],
                         'male hyperoxic mystery': ['Eif1', 'Tuba1c', 'Emd']

                         }
            sc.pl.dotplot(lin_adata, weird_cells, groupby=f"leiden_{lineage}", show=False,
                          save='weird_cells')
        lin_adata.obs["Cell Subtype"] = [leiden_ct_dict[lineage][x] for x in lin_adata.obs[f"leiden_{lineage}"]]
        lin_adata = lin_adata[(~lin_adata.obs["Cell Subtype"].str.startswith('doublet')) & (~lin_adata.obs["Cell Subtype"].str.startswith('low-quality'))]
        # if lineage=='Immune':
        #
        #     sc.tl.umap(lin_adata,min_dist=0.1)
        # else:
        sc.tl.umap(lin_adata, min_dist=0.5)
        sc.pl.umap(lin_adata, color = ['Treatment','Library','leiden',f"leiden_{lineage}",'celltype_rough',"Cell Subtype"],wspace=0.5,show=False,save='_posttrim_leiden')
        sc.pl.dotplot(lin_adata,gene_dict[lineage.lower()],groupby="Cell Subtype",show=False,save='useful_genes_celltype')
        sc.pl.umap(lin_adata, color = gene_dict[lineage.lower()],wspace=0.5,show=False,save='_posttrim_genes')
        sc.pl.umap(lin_adata, color = ['log1p_total_umis','log1p_n_genes_by_umis','doublet_score','ambient_score'],wspace=0.5,show=False,save='_posttrim_qc')
        sc.tl.rank_genes_groups(lin_adata, groupby="Cell Subtype", method='wilcoxon', pts=True)
        sc.tl.dendrogram(lin_adata,"Cell Subtype")
        sc.pl.rank_genes_groups_dotplot(
        lin_adata,
        groupby = "Cell Subtype",
        show = False,
        save = f"celltype_markers.png",
        )
        with pd.ExcelWriter(
            f"{figures_lin}/{lineage}_celltype_markers.xlsx", engine = "xlsxwriter"
        ) as writer:
            for ct in lin_adata.obs["Cell Subtype"].cat.categories:
                df = sc.get.rank_genes_groups_df(lin_adata, key="rank_genes_groups", group=ct)
                df.set_index("names")
                df["pct_difference"] = df["pct_nz_group"] - df["pct_nz_reference"]
                df.to_excel(writer, sheet_name=f"{ct} v rest"[:31])
        # Add Lineage umaps and leiden clusters to top level
        adata.obs[f"umap_{lineage}_1"] = np.nan
        adata.obs[f"umap_{lineage}_2"] = np.nan
        lin_adata.obs[f"umap_{lineage}_1"] = [x[0] for x in lin_adata.obsm["X_umap"]]
        lin_adata.obs[f"umap_{lineage}_2"] = [x[1] for x in lin_adata.obsm["X_umap"]]
        adata.obs[f"umap_{lineage}_1"].loc[lin_adata.obs.index] = lin_adata.obs[
            f"umap_{lineage}_1"
        ]
        adata.obs[f"umap_{lineage}_2"].loc[lin_adata.obs.index] = lin_adata.obs[
            f"umap_{lineage}_2"
        ]
        adata.obs[f"leiden_{lineage}"] = np.nan
        adata.obs[f"leiden_{lineage}"].loc[lin_adata.obs.index] = lin_adata.obs[
            f"leiden_{lineage}"
        ]
        adata.obsm[f"X_umap_{lineage}"] = adata.obs[
            [f"umap_{lineage}_1", f"umap_{lineage}_2"]
        ].to_numpy()
        del adata.obs[f"umap_{lineage}_1"]
        del adata.obs[f"umap_{lineage}_2"]
        adata.obs["Cell Subtype"].loc[lin_adata.obs.index] = lin_adata.obs[
            "Cell Subtype"
        ]
        plot_obs_abundance(lin_adata,'Cell Subtype',hue='Library',ordered=True,
                       as_percentage=True,save=f"{figures_lin}/{lineage}_celltype_abundance.png",hue_order=['N_WT','N_KO','H_WT','H_KO'])
    adata = adata[~adata.obs['Cell Subtype'].isna()]
    ct_order = []
    for lin in adata.obs['Lineage'].cat.categories:
        for ct in sorted(adata[adata.obs['Lineage'] == lin].obs['Cell Subtype'].unique()):
            ct_order.append(ct)
    sc.tl.umap(adata,min_dist=0.5)
    adata.obs['Cell Subtype'] = pd.Categorical(adata.obs['Cell Subtype'], categories=ct_order)
    sc.settings.figdir = figures
    adata.uns['Cell Subtype_colors'] = sc.pl.palettes.default_102.copy()
    sc.pl.umap(adata,color='Cell Subtype',save='Cell_Subtype',show=False)
    plot_obs_abundance(adata, 'Cell Subtype', hue='Library', ordered=True,
                       as_percentage=True, save=f"{figures}/celltype_abundance.png",hue_order=['N_WT','N_KO','H_WT','H_KO'])
    with pd.ExcelWriter(
        f"{figures}/celltype_counts.xlsx", engine="xlsxwriter"
    ) as writer:
        obs_list = ["Library", "Treatment", "Cell Subtype"]
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
    print(adata)
    adata.write(
        f"{data}/{adata_name}_celltyped.gz.h5ad", compression="gzip"
    )


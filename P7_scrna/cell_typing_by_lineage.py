"""Goal:Cell typing by lineage
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
import string
from gtfparse import read_gtf
import anndata
from collections import defaultdict
from functions import plot_obs_abundance
#we set hardcoded paths here
data = "data/single_cell_files/scanpy_files"
adata_name = "kdr_ko_p7_sc"
figures = "data/figures/cell_typing"
os.makedirs(figures, exist_ok=True)
sc.settings.figdir = figures
sc.set_figure_params(dpi=300, format="png")

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
            "Eln","Eng",
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
            'Krt8',
            'Krt18',
            "Ascl1",
            "Calca",
            "Grp",
            "Nrxn1",
            "Ascl2",
            "Pou2f3",
            "Il23a",
            "Foxi1",
            "Ascl3",
            "Cftr",
            "Trp63",  # MGI-approved mouse symbol (not Tp63)
            "Krt5",
            "Ngfr",
            "Scgb1a1",
            "Scube2",
            "Bpifb1",
            "Lcn2",
            "Tspan8",
            "Scgb3a2",
            "Klk11",
            "Mgp",
            "Sox4",
            "Muc5ac",
            "Spdef",
            "Pcdh7",
            "Slc4a11",
            "Agr2",
            "Muc5b",
            "Bpifb2",
            "Lyz2",  # Mouse lysozyme (not LYZ)
            "Ltf",
            "Lpo",
            "Krt14",
            "Epcam",
            "Myh11",
            "Foxj1",
            "Rsph1",
            "Cdhr3",
            "Cdhr4",
            "Deup1",
            "Foxn4",
            "Cdc20b",
            "Sox9",
            "Aldh1a3",
            "Mia1",  # Mouse gene is Mia1
            "Rarres1",
            "Krt19",
            "Serpinb4a",  # Mouse ortholog of human SERPINB4
            "Notch3",
            "Ager",
            "Rtkn2",
            "Sema3b",
            "Lamp3",
            "Abca3",
            "Kcnj15",

        ],
    }
leiden_ct_dict = {
    # "Mesenchymal": {
    #     "0": "Alveolar myofibroblast",
    #     "1": "Adventitial fibroblast",
    #     "2": "Alveolar fibroblast",
    #     "3": "Alveolar fibroblast",
    #     "4": "Alveolar fibroblast",
    #     "5": "Ductal myofibroblast",
    #     "6": "Alveolar fibroblast",
    #     "7": "Pericyte",
    #     "8": "Ductal myofibroblast",
    #     "9": "Adventitial fibroblast",
    #     "10": "Alveolar myofibroblast",
    #     "11": "Ductal myofibroblast",
    #     "12": "Mesothelial",
    #     "13": "Vascular smooth muscle",
    #     "14": "Proliferating myofibroblast",
    #     "15": "Adventitial fibroblast",
    #     "16": "doublet-endo",
    #     "17": "Airway smooth muscle",
    #     "18": "doublet-endo",
    #     "19": "Proliferating fibroblast",
    #     "20": "doublet-endo",
    #     "21": "Proliferating mural",
    # "22": "doublet-endo",
    # "23": "Vascular smooth muscle",
    # },
    "Mesenchymal": {
        "0": "Alveolar fibroblast",
        "1": "Adventitial fibroblast",
        "2": "Alveolar fibroblast",
        "3": "Ductal myofibroblast",
        "4": "Ductal myofibroblast",
        "5": "Alveolar myofibroblast",
        "6": "Alveolar fibroblast",
        "7": "Alveolar myofibroblast",
        "8": "Alveolar myofibroblast",
        "9": "Pericyte",
        "10": "Alveolar fibroblast",
        "11": "Vascular smooth muscle",
        "12": "Mesothelial",
        "13": "Vascular smooth muscle",
        "14": "Alveolar fibroblast",
        "15": "Pericyte",
        "16": "Vascular smooth muscle",
        "17": "Proliferating myofibroblast",
        "18": "Adventitial fibroblast",
        "19": "doublet-Pericyte_endo",
        "20": "doublet-myofibroblast_endo",
        "21": "Pericyte",
        "22": "Airway smooth muscle",
        "23": "Proliferating fibroblast",
        "24": "Proliferating mural",
        "25": "doublet-Adventitial fibroblast_endo",
        "26": "doublet-Alveolar fibroblast_endo",
    },
    "Endothelial": {
        "0": "Cap1",
        "1": "Cap1",
        "2": "Cap2",
        "3": "Cap1",
        "4": "Cap1",
        "5": "Cap1_Cap2",
        "6": "Cap1",
        "7": "Prolfierating EC",
        "8": "Arterial EC",
        "9":  "Cap1_Cap2",
        "10": "Venous EC",
        "11": "Cap1_Cap2",
        "12": "Cap2",
        "13": "Lymphatic EC",
        "14": "Systemic Venous EC",
        "15": "doublet-epi",
    },
    "Immune": {
        "0": "Basophil",
        "1": "Alveolar macrophage",
        "2": "B cell",
        "3": "B cell",
        "4": "Basophil",
        "5": "Monocyte",
        "6": "B cell",
        "7": "mig-DC",
        "8": "doublet",
        "9":"Proliferating monocyte",
        "10":  "Monocyte",
        "11": "T cell",
        "12": "Neutrophil",
        "13": "doublet",
        "14": "Proliferating Alveolar macrophage",


    },
    "Epithelial": {
        "0": "AT1",
        "1": "AT2",
        "2": "AT2",
        "3": "AT2",
        "4": "AT2",
        "5": "AT1",
        "6": "AT1",
        "7": "Ciliated",
        "8": "doublet-endo",
        "9": "doublet-endo",
        "10": "doublet-endo",
        "11": "AT1",
        "12": "Neuroendocrine_Other_Epithelial",
        "13": "AT1_AT2",
        "14": "Proliferating AT2",
        "15": "doublet-endo",
        "16": "Club",
        "17": "doublet-mese",
        "18": "doublet-endo",

    },

}
if __name__ == "__main__":
    # adata = sc.read(
    #     f"{data}/{adata_name}_filtered_embed_clean.gz.h5ad",
    # )
    adata = sc.read(
        f"{data}/{adata_name}_filtered_embed_cleaned.gz.h5ad",
    )
    print(adata)
    adata.obs["Cell Subtype"] = pd.Series(index=adata.obs.index, data=None, dtype="str")
    for lineage in adata.obs['Lineage'].cat.categories:
    # for lineage in ['Mesenchymal']:
        gene_ls = [x for x in gene_dict[lineage.lower()] if x in adata.var_names]
        figures_lin = f"data/figures/cell_typing/{lineage}"
        os.makedirs(figures_lin, exist_ok=True)
        sc.settings.figdir = figures_lin
        print(lineage)
        lin_adata = adata[adata.obs['Lineage'] == lineage]
        sc.pp.highly_variable_genes(lin_adata, batch_key="Library")
        sc.pp.pca(lin_adata, mask_var="highly_variable")
        sc.pp.neighbors(lin_adata, use_rep="X_pca")
        if lineage=='Mesenchymal':
            sc.tl.leiden(lin_adata, key_added=f"leiden_{lineage}", resolution=1.2)
        else:
            sc.tl.leiden(lin_adata, key_added=f"leiden_{lineage}", resolution=1)
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
        sc.pl.umap(lin_adata, color = ['leiden',f"leiden_{lineage}",'Library',
                                       'celltype_rough'
                                       ],wspace=0.5,show=False,save='_pretrim_leiden')
        sc.pl.dotplot(lin_adata,gene_ls,groupby=f"leiden_{lineage}",show=False,save='useful_genes_leiden')
        sc.pl.umap(lin_adata, color = gene_ls,wspace=0.5,show=False,save='_pretrim_genes')
        sc.pl.umap(lin_adata, color = ['log1p_total_umis','log1p_n_genes_by_umis','ambient_score','doublet_score'],wspace=0.5,show=False,save='_pretrim_qc')
        if lineage == 'Mesenchymal':
            weird_cells = {'striated_muscle': ['Ttn', 'Ryr2', 'Myh6', 'Tbx20', 'Ldb3'],
                         'multi_ab_musc': ['Gm20754', 'Pdzrn4', 'Chrm2', 'Cacna2d3', 'Chrm3'],
                         'multi_acta1': ['Eya4', 'Rbm20', 'Neb', 'Itm2a'],
                         'male hyperoxic fibroblast': [ 'Actc1', 'Tuba4a'],
                         'male hyperoxic mystery': ['Eif1', 'Tuba1c', 'Emd']

                         }
            sc.pl.dotplot(lin_adata, weird_cells, groupby=f"leiden_{lineage}", show=False,
                          save='weird_cells')
        lin_adata.obs["Cell Subtype"] = [leiden_ct_dict[lineage][x] if x in leiden_ct_dict[lineage].keys() else x for x in lin_adata.obs[f"leiden_{lineage}"]]
        lin_adata = lin_adata[(~lin_adata.obs["Cell Subtype"].str.startswith('doublet')) & (~lin_adata.obs["Cell Subtype"].str.startswith('low-quality'))]
        if lineage=='Immune':

            sc.tl.umap(lin_adata,min_dist=0.1)
        else:
            sc.tl.umap(lin_adata, min_dist=0.5)
        sc.pl.umap(lin_adata, color = ['Treatment','Library','leiden',f"leiden_{lineage}",
                                       'celltype_rough','Cell Subtype'
                                       ],wspace=0.5,show=False,save='_posttrim_leiden')
        sc.pl.dotplot(lin_adata,gene_ls,groupby="Cell Subtype",show=False,save='useful_genes_celltype')
        sc.pl.umap(lin_adata, color = gene_ls,wspace=0.5,show=False,save='_posttrim_genes')
        sc.pl.umap(lin_adata, color = ['log1p_total_umis','log1p_n_genes_by_umis','ambient_score','doublet_score'],wspace=0.5,show=False,save='_posttrim_qc')
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
        with pd.ExcelWriter(
            f"{figures_lin}/metadata_counts.xlsx", engine="xlsxwriter"
        ) as writer:
            obs_list = ["Library", f"leiden_{lineage}","Cell Subtype"]
            num_obs = len(obs_list) + 1
            for ind in range(0, num_obs):
                for subset in itertools.combinations(obs_list, ind):
                    if len(subset) != 0:
                        subset = list(subset)
                        if len(subset) == 1:
                            key = subset[0]
                            lin_adata.obs[key].value_counts().to_excel(writer, sheet_name=key)
                        else:
                            key = "_".join(subset)
                            lin_adata.obs.groupby(subset[:-1])[subset[-1]].value_counts(
                                normalize=True
                            ).to_excel(writer, sheet_name=key[:31])
    adata = adata[~adata.obs['Cell Subtype'].isna()]
    ct_order = []
    for lin in adata.obs['Lineage'].cat.categories:
        for ct in sorted(adata[adata.obs['Lineage'] == lin].obs['Cell Subtype'].unique()):
            ct_order.append(ct)
    sc.tl.umap(adata,min_dist=0.5)
    adata.obs['Cell Subtype'] = pd.Categorical(adata.obs['Cell Subtype'], categories=ct_order)
    sc.settings.figdir = figures
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
    adata.write(
        f"{data}/{adata_name}_celltyped.gz.h5ad", compression="gzip"
    )


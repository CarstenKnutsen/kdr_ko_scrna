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
figures = "data/figures/cell_typing_harmony_clean"
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
    }
leiden_ct_dict = {
    "Mesenchymal": {
        "0": "Alveolar fibroblast",
        "1": "Alveolar fibroblast",
        "2": "Adventitial fibroblast",
        "3": "Ductal myofibroblast",
        "4": "Alveolar myofibroblast",
        "5": "Pericyte",
        "6": "Airway smooth muscle_myofibroblast combined",
        "7": "Alveolar Myofibroblast",
        "8": "Adventitial fibroblast",
        "9": "Alveolar fibroblast",
        "10": "Airway smooth muscle",
        "11": "Mesothelial",
        "12": "Vascular smooth muscle",
        "13": "Vascular smooth muscle",
        "14": "Vascular smooth muscle",
        "15": "doublet-Airway smooth muscle",
        "16": "Proliferating myofibroblast",
        "17": "Proliferating fibroblast",
        "18": "Pericyte",
        "19": "Cardiomyocyte",
        "20": "Vascular smooth muscle",
    },
    "Endothelial": {
        "0": "Cap1",
        "1": "Cap1",
        "2": "Cap2",
        "3": "Cap2",
        "4": "Cap1_Cap2",
        "5": "Cap1",
        "6": "Cap1",
        "7": "Cap1",
        "8": "Venous EC",
        "9":  "Proliferating EC",
        "10": "Cap1_Cap2",
        "11": "Arterial EC",
        "12": "Cap2",
        "13": "Lymphatic EC",
    },
    "Immune": {
        "0": "Basophil",
        "1": "Basophil",
        "2": "Basophil",
        "3": "Basophil",
        "4": "doublet-endo",
        "5": "Basophil",

    },
    "Epithelial": {
        "0": "AT1",
        "1": "AT2",
        "2": "AT1",
        "3": "AT2",
        "4": "AT1",
        "5": "AT2",
        "6": "AT2",
        "7": "Ciliated",
        "8": "AT2",
        "9": "AT1",
        "10": "doublet-endo_AT2",
        "11": "AT2",
        "12": "doublet-endo_AT1",
        "13": "doublet-endo_AT2",
        "14": "AT1_AT2",
        "15": "AT2",
        "16": "AT1",
        "17": "AT2",
        "18": "Proliferating AT2",
        "19": "Ciliated",
        "20": "doublet-mese_AT2",
        "21": "doublet-mese_AT1",

    },

}
if __name__ == "__main__":
    adata = sc.read(
        f"{data}/{adata_name}_filtered_embed_clean.gz.h5ad",
    )
    print(adata)
    adata.obs["Cell Subtype"] = pd.Series(index=adata.obs.index, data=None, dtype="str")
    for lineage in adata.obs['Lineage'].cat.categories:
        figures_lin = (f""
                       f"{figures}/{lineage}")
        os.makedirs(figures_lin, exist_ok=True)
        sc.settings.figdir = figures_lin
        print(lineage)
        lin_adata = adata[adata.obs['Lineage'] == lineage]
        sc.pp.highly_variable_genes(lin_adata, batch_key="Library")
        sc.pp.pca(lin_adata, mask_var="highly_variable")
        sce.pp.harmony_integrate(lin_adata, 'Library',adjusted_basis='X_pca')

        # sc.pp.neighbors(lin_adata, use_rep="X_pca")
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
        sc.pl.dotplot(lin_adata,gene_dict[lineage.lower()],groupby=f"leiden_{lineage}",show=False,save='useful_genes_leiden')
        sc.pl.umap(lin_adata, color = gene_dict[lineage.lower()],wspace=0.5,show=False,save='_pretrim_genes')
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
        sc.pl.dotplot(lin_adata,gene_dict[lineage.lower()],groupby="Cell Subtype",show=False,save='useful_genes_celltype')
        sc.pl.umap(lin_adata, color = gene_dict[lineage.lower()],wspace=0.5,show=False,save='_posttrim_genes')
        sc.pl.umap(lin_adata, color = ['log1p_total_umis','log1p_n_genes_by_umis','ambient_score','doublet_score'],wspace=0.5,show=False,save='_posttrim_qc')
        sc.tl.rank_genes_groups(lin_adata, groupby="Cell Subtype", method='wilcoxon', pts=True)
        if lineage != 'Immune':

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
            obs_list = ["Library", "leiden", "celltype_rough"]
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
    # adata = adata[~adata.obs['Cell Subtype'].isna()]
    # ct_order = []
    # for lin in adata.obs['Lineage'].cat.categories:
    #     for ct in sorted(adata[adata.obs['Lineage'] == lin].obs['Cell Subtype'].unique()):
    #         ct_order.append(ct)
    # sc.tl.umap(adata,min_dist=0.5)
    # adata.obs['Cell Subtype'] = pd.Categorical(adata.obs['Cell Subtype'], categories=ct_order)
    # sc.settings.figdir = figures
    # sc.pl.umap(adata,color='Cell Subtype',save='Cell_Subtype',show=False)
    # plot_obs_abundance(adata, 'Cell Subtype', hue='Library', ordered=True,
    #                    as_percentage=True, save=f"{figures}/celltype_abundance.png",hue_order=['N_WT','N_KO','H_WT','H_KO'])
    # with pd.ExcelWriter(
    #     f"{figures}/celltype_counts.xlsx", engine="xlsxwriter"
    # ) as writer:
    #     obs_list = ["Library", "Treatment", "Cell Subtype"]
    #     num_obs = len(obs_list) + 1
    #     for ind in range(0, num_obs):
    #         for subset in itertools.combinations(obs_list, ind):
    #             if len(subset) != 0:
    #                 subset = list(subset)
    #                 if len(subset) == 1:
    #                     key = subset[0]
    #                     adata.obs[key].value_counts().to_excel(writer, sheet_name=key)
    #                 else:
    #                     key = "_".join(subset)
    #                     adata.obs.groupby(subset[:-1])[subset[-1]].value_counts(
    #                         normalize=True
    #                     ).to_excel(writer, sheet_name=key[:31])
    # adata.write(
    #     f"{data}/{adata_name}_celltyped.gz.h5ad", compression="gzip"
    # )


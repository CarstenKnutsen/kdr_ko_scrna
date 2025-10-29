"""Goal:Cell typing by lineage with the cell cycle genes regressed out
Date:250204
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

from scipy import sparse
#we set hardcoded paths here
data = "data/single_cell_files/scanpy_files"
adata_name = "kdr_ko_p7"
figures = "data/figures/cell_typing_no_cc"
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
            "Myh11",
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
        "1": "Adventitial fibroblast",
        "2": "Mesothelial",
        "3": "Airway smooth muscle",
        "4": "Pericyte",
        "5": "Vascular smooth muscle",
        "6": "Myofibroblast",
        "7": "low-quality_Fibroblast",
        "8":"Adventitial fibroblast",
        "9": "Schwann cell",
        "10":  "Alveolar fibroblast",
        "11":"doublet_Epithelial",
        "12":"low-quality Pericyte",
        "13":"Mesothelial",
        "14": "Mesothelial",
        "15": "Aberrant muscle",
        "16":"Striated muscle",
        "17": "Dsc2_3+ cell",
        "18": "Airway smooth muscle"

    },
    "Endothelial": {
        "0": "Cap1",
        "1": "Cap1",
        "2": "Cap1",
        "3": "Cap2",
        "4": "Cap1_Cap2",
        "5": "Cap1_Cap2",
        "6": "Cap1",
        "7": "Arterial EC",
        "8":  "Venous EC",
        "9":  "Cap1",
        "10": "Lymphatic EC",
        "11": "Cap1",
        "12": "Cap1",
        "13": "Cap2",
        "14": "Cap1",
        "15": "Cap1_Cap2",
        "16": "Venous EC",
        "17": "Arterial EC",
        "18": "Cap1_Cap2",
        "19": "Cap2",
    },
    "Immune": {
        "0": "Alveolar macrophage",
        "1": "Alveolar macrophage",
        "2": "Alveolar macrophage",
        "3": "B cell",
        "4": "doublet_endothelial",
        "5": "Monocyte",
        "6": "Alveolar macrophage",
        "7": "Mast cell",
        "8": "Alveolar macrophage",
        "9": "Interstitial macrophage",
        "10":  "Alveolar macrophage",
        "11": "B cell",
        "12":"Alveolar macrophage",
        "13": "Basophil",
        "14": "T cell",
        "15": "c-Dendritic cell",
        "16": "mig-Dendritic cell",
    },
    "Epithelial": {
        "0": "AT1_AT2",
        "1": "Ciliated",
        "2": "AT2",
        "3": "AT1",
        "4": "Ciliated",
        "5": "AT2",
        "6": "Club",
        "7": "Goblet",
        "8": "doublet_epithelial cell",
        "9": "AT1",
        "10": "Ciliated",
        "11": "Neuroendocrine",
        "12": "Ciliated",

    },

}
if __name__ == "__main__":
    adata = sc.read(
        f"{data}/{adata_name}_celltyped.gz.h5ad", compression="gzip",
    )
    print(adata)
    cell_cycle_genes = [x.strip() for x in open('data/outside_data/regev_lab_cell_cycle_genes.txt')]
    cell_cycle_genes = [x.lower().capitalize() for x in cell_cycle_genes]
    s_genes = cell_cycle_genes[:43]
    g2m_genes = cell_cycle_genes[43:]
    cell_cycle_genes = [x for x in cell_cycle_genes if x in adata.var_names]
    sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)
    sc.tl.score_genes(adata, ['Mki67', 'Top2a', 'Birc5', 'Hmgb2', 'Cenpf'], score_name='proliferation_score')
    sc.pp.regress_out(adata, ['S_score', 'G2M_score'])
    adata.layers['log1p_cc_regress'] = sparse.csr_matrix(adata.X).copy()
    adata.obs["Cell Subtype_no_cc"] = pd.Series(index=adata.obs.index, data=None, dtype="str")
    for lineage in adata.obs['Lineage'].cat.categories:
        figures_lin = f"{figures}/{lineage}"
        os.makedirs(figures_lin, exist_ok=True)
        sc.settings.figdir = figures_lin
        print(lineage)
        lin_adata = adata[adata.obs['Lineage'] == lineage]
        sc.pp.highly_variable_genes(lin_adata, batch_key="Library")
        sc.pp.pca(lin_adata, mask_var="highly_variable")
        sc.pp.neighbors(lin_adata, use_rep="X_pca")
        sc.tl.leiden(lin_adata, key_added=f"leiden_{lineage}_no_cc", resolution=1)
        sc.tl.dendrogram(lin_adata,f"leiden_{lineage}")
        lin_adata.X = lin_adata.layers['log1p'].copy()
        sc.tl.rank_genes_groups(lin_adata, groupby=f"leiden_{lineage}_no_cc",method='wilcoxon',pts=True)
        sc.pl.rank_genes_groups_dotplot(
            lin_adata,
            groupby=f"leiden_{lineage}_no_cc",
            show=False,
            save=f"leiden_markers.png",
        )
        with pd.ExcelWriter(
                f"{figures_lin}/{lineage}_leiden_markers.xlsx", engine="xlsxwriter"
        ) as writer:
            for ct in lin_adata.obs[f"leiden_{lineage}_no_cc"].cat.categories:
                df = sc.get.rank_genes_groups_df(lin_adata, key="rank_genes_groups", group=ct)
                df.set_index("names")
                df["pct_difference"] = df["pct_nz_group"] - df["pct_nz_reference"]
                df.to_excel(writer, sheet_name=f"{ct} v rest"[:31])
        if lineage == 'Mesenchymal':
            weird_cells = {'striated_muscle': ['Ttn', 'Ryr2', 'Myh6', 'Tbx20', 'Ldb3'],
                         'multi_ab_musc': ['Gm20754', 'Pdzrn4', 'Chrm2', 'Cacna2d3', 'Chrm3'],
                         'multi_acta1': ['Eya4', 'Rbm20', 'Neb', 'Itm2a'],
                         'male hyperoxic fibroblast': ['Actc1', 'Tuba4a'],
                         'male hyperoxic mystery': ['Eif1', 'Tuba1c', 'Emd']

                         }
            sc.pl.dotplot(lin_adata, weird_cells, groupby=f"leiden_{lineage}_no_cc", show=False,
                          save='weird_cells')
        # if lineage == 'Endothelial':
        #
        #     lin_adata.obs["Cell Subtype_no_cc"] = [leiden_ct_dict[lineage][x] for x in lin_adata.obs[f"leiden_{lineage}_no_cc"]]
        # else:
        #     lin_adata.obs["Cell Subtype_no_cc"] = lin_adata.obs["Cell Subtype"]
        # lin_adata.obs["Cell Subtype_no_cc"] = [leiden_ct_dict[lineage][x] for x in
        #                                        lin_adata.obs[f"leiden_{lineage}_no_cc"]]
        result = {group: values['Cell Subtype'][~values['Cell Subtype'].str.startswith("Proliferating")].value_counts().idxmax()
                  for group, values in lin_adata.obs.groupby(f'leiden_{lineage}_no_cc')}
        lin_adata.obs["Cell Subtype_no_cc"] = [result[x] for x in
                                               lin_adata.obs[f"leiden_{lineage}_no_cc"]]
        if lineage=='Mesenchymal':
            lin_adata.obs["Cell Subtype_no_cc"] = [x if y!='9' else 'Adventitial fibroblast'for x,y in zip(lin_adata.obs['Cell Subtype_no_cc'],lin_adata.obs[f'leiden_{lineage}_no_cc'])]

        # lin_adata = lin_adata[(~lin_adata.obs["Cell Subtype_no_cc"].str.startswith('doublet')) & (~lin_adata.obs["Cell Subtype_no_cc"].str.startswith('low-quality'))]
        sc.tl.umap(lin_adata,min_dist=0.5)
        sc.pl.umap(lin_adata, color = ['Treatment','Library','leiden',f"leiden_{lineage}",f"leiden_{lineage}_no_cc",'celltype_rough',"Cell Subtype","Cell Subtype_no_cc"],wspace=0.5,show=False,save='_posttrim_leiden')
        sc.pl.dotplot(lin_adata,gene_dict[lineage.lower()],groupby="Cell Subtype",show=False,save='useful_genes_celltype')
        sc.pl.dotplot(lin_adata,gene_dict[lineage.lower()],groupby=f"leiden_{lineage}_no_cc",show=False,save='useful_genes_leiden')
        sc.pl.umap(lin_adata, color = gene_dict[lineage.lower()],wspace=0.5,show=False,save='_posttrim_genes')
        sc.pl.umap(lin_adata, color = ['log1p_total_umis','log1p_n_genes_by_umis'],wspace=0.5,show=False,save='_posttrim_qc')
        sc.pl.umap(lin_adata, color = ['phase','proliferation_score'],wspace=0.5,show=False,save='_posttrim_cc')
        sc.tl.rank_genes_groups(lin_adata, groupby="Cell Subtype_no_cc", method='wilcoxon', pts=True)
        sc.tl.dendrogram(lin_adata,"Cell Subtype_no_cc")
        sc.pl.rank_genes_groups_dotplot(
        lin_adata,
        groupby = "Cell Subtype_no_cc",
        show = False,
        save = f"celltype_markers.png",
        )
        with pd.ExcelWriter(
            f"{figures_lin}/{lineage}_celltype_markers.xlsx", engine = "xlsxwriter"
        ) as writer:
            for ct in lin_adata.obs["Cell Subtype_no_cc"].cat.categories:
                df = sc.get.rank_genes_groups_df(lin_adata, key="rank_genes_groups", group=ct)
                df.set_index("names")
                df["pct_difference"] = df["pct_nz_group"] - df["pct_nz_reference"]
                df.to_excel(writer, sheet_name=f"{ct} v rest"[:31])
        # Add Lineage umaps and leiden clusters to top level
        adata.obs[f"umap_{lineage}_no_cc_1"] = np.nan
        adata.obs[f"umap_{lineage}_no_cc_2"] = np.nan
        lin_adata.obs[f"umap_{lineage}_no_cc_1"] = [x[0] for x in lin_adata.obsm["X_umap"]]
        lin_adata.obs[f"umap_{lineage}_no_cc_2"] = [x[1] for x in lin_adata.obsm["X_umap"]]
        adata.obs[f"umap_{lineage}_no_cc_1"].loc[lin_adata.obs.index] = lin_adata.obs[
            f"umap_{lineage}_no_cc_1"
        ]
        adata.obs[f"umap_{lineage}_no_cc_2"].loc[lin_adata.obs.index] = lin_adata.obs[
            f"umap_{lineage}_no_cc_2"
        ]
        adata.obs[f"leiden_{lineage}_no_cc"] = np.nan
        adata.obs[f"leiden_{lineage}_no_cc"].loc[lin_adata.obs.index] = lin_adata.obs[
            f"leiden_{lineage}_no_cc"
        ]
        adata.obsm[f"X_umap_{lineage}_no_cc"] = adata.obs[
            [f"umap_{lineage}_no_cc_1", f"umap_{lineage}_no_cc_2"]
        ].to_numpy()
        del adata.obs[f"umap_{lineage}_no_cc_1"]
        del adata.obs[f"umap_{lineage}_no_cc_2"]
        adata.obs["Cell Subtype_no_cc"].loc[lin_adata.obs.index] = lin_adata.obs[
            "Cell Subtype_no_cc"
        ]
        plot_obs_abundance(lin_adata,'Cell Subtype_no_cc',hue='Library',ordered=True,
                       as_percentage=True,save=f"{figures_lin}/{lineage}_celltype_abundance.png",hue_order=['N_WT','N_KO','H_WT','H_KO'])
    adata = adata[~adata.obs['Cell Subtype_no_cc'].isna()]
    ct_order = []
    for lin in adata.obs['Lineage'].cat.categories:
        for ct in sorted(adata[adata.obs['Lineage'] == lin].obs['Cell Subtype_no_cc'].unique()):
            ct_order.append(ct)
    sc.tl.umap(adata,min_dist=0.5)
    adata.obs['Cell Subtype_no_cc'] = pd.Categorical(adata.obs['Cell Subtype_no_cc'], categories=ct_order)
    sc.settings.figdir = figures
    sc.pl.umap(adata,color='Cell Subtype_no_cc',save='Cell_Subtype',show=False)
    plot_obs_abundance(adata, 'Cell Subtype_no_cc', hue='Treatment', ordered=True,
                       as_percentage=True, save=f"{figures}/celltype_abundance.png",hue_order=['Normoxia','Hyperoxia'])
    with pd.ExcelWriter(
        f"{figures}/celltype_counts.xlsx", engine="xlsxwriter"
    ) as writer:
        obs_list = ["Library", "Treatment", "Cell Subtype_no_cc"]
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

    color_dict={}
    for ind,ct in enumerate(adata.obs['Cell Subtype'].cat.categories):
        color_dict[ct] = adata.uns['Cell Subtype_colors'][ind]
    new_colors = []
    for ct in adata.obs['Cell Subtype_no_cc'].cat.categories:
        new_colors.append(color_dict[ct])
    adata.uns['Cell Subtype_no_cc_colors'] = new_colors
    adata.X = adata.layers['log1p'].copy()
    adata.write(
        f"{data}/{adata_name}_celltyped_no_cc.gz.h5ad", compression="gzip"
    )


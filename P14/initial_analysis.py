"""Goal:Initial analysis of P14 Kdr KO scRNAseq, creating h5ad, filtering, and embedding
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
from gtfparse import read_gtf
import anndata
from collections import defaultdict

#we set hardcoded paths here
gtf_fn = "data/genes.gtf"
# input_data = "data/single_cell_files/cellranger_output"
input_data = "data/single_cell_files/soupx"

output_data = "data/single_cell_files/scanpy_files"
os.makedirs(output_data, exist_ok=True)
adata_name = "kdr_ko_p14"
figures = "data/figures"
os.makedirs(figures, exist_ok=True)
sc.set_figure_params(dpi=300, format="png")
sc.settings.figdir = figures
#usefula gene lists below
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
            "Cdh5",
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
#leiden dictionary to assign cell types
leiden_ct_dict = {
    "0": "low-quality",
    "1": "AT2",
    "2": "AT1",
    "3": "Cap1",
    "4": "Alveolar fibroblast",
    "5": "Cap2",
    "6": "Myeloid",
    "7": "ASM",
    "8": "Ciliated",
    "9": "VSM",
    "10":  "Pericyte",
    "11":"Leukocyte",
    "12": "Mesothelial",
    "13": "Upper airway epithelial",
    "14": "doublet-all ",
    "15": "doublet-endothelial",
    "16": "Leukocyte",
    "17": "Lymphatic EC",
    "18": "Neuroendocrine",

}
def read_adata(folder):
    adata = sc.read_mtx(f'{folder}/matrix.mtx').T
    features = pd.read_csv(f'{folder}/genes.tsv',
                          sep = '\t',
                          header = None)
    bc = pd.read_csv(f'{folder}/barcodes.tsv',
                          sep = '\t',
                    header = None)
    features.rename(columns={0:'gene_id',
                            1: 'gene_symbol',
                            2: 'category'},
                   inplace = True)

    adata.var = features
    adata.obs_names = bc[0]
    adata.var_names = adata.var['gene_id'].values
    return adata
if __name__ == "__main__":
    ## Here we are reading in cellranger outputs and concatentating together, adding some metadata for both cells in obs and genes in var
    runs = os.listdir(input_data)
    adatas = []
    gtf = read_gtf(gtf_fn).to_pandas()
    print(gtf.columns)
    gene_name_dict = pd.Series(gtf['gene_name'].values, index=gtf['gene_id']).to_dict()
    for x in gene_name_dict.keys():
        if gene_name_dict[x] == '':
            gene_name_dict[x] = x
    ambient_dict = {}
    for run in runs:
        print(run)
        if run == "Undetermined":
            continue

        folder = f"{input_data}/{run}"
        ambient_df = pd.read_csv(f'{figures}/soupx/{run}_ambient_genes.txt', sep='\t', header=0, index_col=0)
        ambient_dict[run] = ambient_df
        adata = read_adata(folder)
        adata2 = sc.read_10x_h5(f"data/single_cell_files/cellranger_output/{run}/outs/filtered_feature_bc_matrix.h5")
        adata.layers['raw'] = adata2.X.copy()
        adata.obs_names = run + "_" + adata.obs_names
        meta = run.split("_")[0]
        adata.obs["Treatment"] =meta[0]
        adata.obs["Timepoint"] = meta[-3:]
        adata.obs["Genotype"] = meta[-5:-3]
        adata.obs["Library"] = adata.obs["Treatment"] + '_' + adata.obs["Genotype"]
        adatas.append(adata.copy())
    adata = anndata.concat(adatas)

    adata.obs["Treatment"].replace({"H": "Hyperoxia", "N": "Normoxia"}, inplace=True)
    adata.obs["Treatment"] = pd.Categorical(adata.obs['Treatment'],categories = ['Normoxia','Hyperoxia'])
    adata.obs["Genotype"] = pd.Categorical(adata.obs['Genotype'],categories = ['WT','KO'])
    adata.obs["Library"] = pd.Categorical(adata.obs['Library'],categories = ['N_WT','N_KO','H_WT','H_KO'])


    adata.var['ambient_rna_est_contamination'] = pd.concat( {key: df['est'] for key, df in ambient_dict.items()}).groupby(level=1).mean()
    adata.var['ambient_rna_total_counts'] = pd.concat({key: df['counts'] for key, df in ambient_dict.items()}).groupby(level=1).sum()
    for key, df in ambient_dict.items():
        adata.var[f'ambient_rna_est_contamination_{key}'] = df['est']
        adata.var[f'ambient_rna_counts_{key}'] = df['counts']
    for column in ["gene_name","gene_id", "gene_type", "seqname", "transcript_name", "protein_id"]:
        temp_dict = pd.Series(gtf[column].values, index=gtf["gene_id"]).to_dict()
        # temp_dict.update(pd.Series(gtf[column].values, index=gtf["gene_id"]).to_dict())
        temp_dict = defaultdict(lambda: None, temp_dict)
        adata.var[column] = [temp_dict[x] for x in adata.var.index]
    adata.var_names = [gene_name_dict[x] for x in adata.var_names]
    adata.var_names_make_unique()
    adata.var["mt"] = [True if x == "chrM" else False for x in adata.var["seqname"]]
    adata.var["ribo"] = adata.var_names.str.startswith(("Rps", "Rpl"))
    adata.var["hb"] = adata.var_names.str.contains(("^Hb[^(p)]"))
    print(adata)
    ## write out unfiltered object here
    adata.write(
        f"{output_data}/{adata_name}_cellranger_all_cells.gz.h5ad", compression="gzip"
    )
    ## Run qc and make some qc graphs
    figures_qc = f"{figures}/qc"
    os.makedirs(figures_qc, exist_ok=True)
    sc.settings.figdir = figures_qc

    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt", "ribo", "hb"],
        expr_type="umis",
        percent_top=[20],
        log1p=True,
        inplace=True,
    )

    sc.pl.scatter(
        adata,
        x="log1p_total_umis",
        y="log1p_n_genes_by_umis",
        show=False,
        save="_genes_by_counts_pretrim_log",
    )
    sc.pl.highest_expr_genes(adata, n_top=20, show=False, save=f"_pretrim")
    sc.pl.violin(
        adata,
        ["log1p_total_umis"],
        groupby="Library",
        rotation=90,
        show=False,
        save="_umis_pretrim",
    )
    sc.pl.violin(
        adata,
        ["log1p_n_genes_by_umis"],
        groupby="Library",
        rotation=90,
        show=False,
        save="_genes_pretrim",
    )
    sc.pl.violin(
        adata,
        ["pct_umis_mt"],
        groupby="Library",
        rotation=90,
        show=False,
        save="_mt_pretrim",
    )
    # We are running scrublet here to detect any doublets on the raw count matrix
    sc.pp.scrublet(adata, batch_key="Library")
    print(adata)
    sc.pl.scrublet_score_distribution(adata, show=False, save="scrublet_scores")
    # instead of filtering with thresholds we are filtering with outlier testing for three categories, level of UMI, level of unique genes, and pct of UMIs in highly expressed genes
    def is_outlier(adata, metric: str, nmads: int):
        M = adata.obs[metric]
        outlier = (M < np.median(M) - nmads * median_abs_deviation(M)) | (
            np.median(M) + nmads * median_abs_deviation(M) < M
        )
        return outlier

    adata.obs["outlier"] = (
        is_outlier(adata, "log1p_total_umis", 5)
        | is_outlier(adata, "log1p_n_genes_by_umis", 5)
        | is_outlier(adata, "pct_umis_in_top_20_genes", 5)
    )  # https://www.sc-best-practices.org/preprocessing_visualization/quality_control.html#
    adata.obs["mt_outlier"] = adata.obs["pct_umis_mt"] > 30
    print(adata.obs.mt_outlier.value_counts())
    print(adata.obs.outlier.value_counts())
    print(adata.obs.predicted_doublet.value_counts())
    adata = adata[(~adata.obs.outlier) & (~adata.obs.mt_outlier)& (~adata.obs.predicted_doublet)].copy()
    #filter out genes expressed in less than 10 cells, arbitrary
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pl.scatter(
        adata,
        x="log1p_total_umis",
        y="log1p_n_genes_by_umis",
        show=False,
        save="_genes_by_umis_posttrim_log",
    )
    sc.pl.highest_expr_genes(adata, n_top=20, show=False, save=f"_posttrim")
    sc.pl.violin(
        adata,
        ["log1p_total_umis"],
        rotation=90,
        show=False,
        groupby="Library",
        save="_umis_posttrim",
    )
    sc.pl.violin(
        adata,
        ["log1p_n_genes_by_umis"],
        rotation=90,
        show=False,
        groupby="Library",
        save="_genes_posttrim",
    )
    sc.pl.violin(
        adata,
        ["pct_umis_mt"],
        groupby="Library",
        rotation=90,
        show=False,
        save="_mt_posttrim",
    )
    print(f"Median UMIS: {adata.obs['total_umis'].median()}")
    print(f"Median Genes: {adata.obs['n_genes_by_umis'].median()}")
    # # run embedding and clustering below
    figures_embed = f"{figures}/initial_embedding"
    os.makedirs(figures_embed, exist_ok=True)
    sc.settings.figdir = figures_embed
    adata.layers["soupx"] = adata.X.copy()
    sc.pp.normalize_total(adata, key_added=None, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["log1p"] = adata.X.copy()
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
    adata.obs['Lineage'] = [lineage_dict[x] for x in adata.obs['leiden']]
    adata.obs["celltype_rough"] = [leiden_ct_dict[x] for x in adata.obs["leiden"]]
    ## make plots below that are helpful for initial analysis
    sc.pl.dotplot(
        adata,
        ['Cdh5','Col1a1','Ptprc','Epcam','Muc1','Muc5b','Foxa2','Cdh1','Krt8','Sox9','Tbx2','Kit','Car4','Sftpc','Hopx'],
        groupby="leiden",
        dendrogram=False,
        show=False,
        save="useful_genes.png",
    )
    for lin, genes in gene_dict.items():
        sc.pl.dotplot(
            adata,
           genes,
            groupby="leiden",
            dendrogram=False,
            show=False,
            save=f"useful_genes_{lin}.png",
        )

    adata.write(
        f"{output_data}/{adata_name}_filtered_embed_w_doublets.gz.h5ad", compression="gzip"
    )
    adata = sc.read(f"{output_data}/{adata_name}_filtered_embed_w_doublets.gz.h5ad")
    adata =adata[(~adata.obs['celltype_rough'].str.startswith('doublet'))&(~adata.obs['celltype_rough'].str.startswith('low-quality'))]
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
        'ambient_score',
        'Cdh5','Col1a1','Ptprc','Epcam','Muc1','Tbx2','Kit','Car4','Sftpc','Hopx'
    ]:
        sc.pl.umap(adata, color=color, show=False, save=f"_{color}.png")
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
        lin_adata = adata[adata.obs['Lineage']==lineage]
        sc.tl.dendrogram(lin_adata,groupby='leiden')
        sc.tl.rank_genes_groups(lin_adata, groupby='leiden')
        sc.pl.rank_genes_groups_dotplot(lin_adata,save=f'{lineage}',show=False)
        sc.pl.umap(lin_adata,color=['leiden',
                                    # 'celltype_rough'
                                    ],save=f'{lineage}',show=False)
    adata.write(
        f"{output_data}/{adata_name}_filtered_embed.gz.h5ad", compression="gzip"
    )

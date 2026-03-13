# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python (buddi-v2)
#     language: python
#     name: buddi-v2
# ---

# %% [markdown]
# # This notebook demonstrates final integration and normalization steps on augmetned single cell reference and bulk data before training buddi4 model from buddi_v2

# %%
import joblib
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from matplotlib_venn import venn2, venn3


# %%
SAMPLE_ID_COL = "sample_id"
STIM_COL = "stim"
GENE_ID_COL = "gene_ids"
TECH_COL = "samp_type"

# %% [markdown]
# ## Add buddi_v2 software to path

# %%
from buddi_v2.data import BuDDINData

# %% [markdown]
# ## Pre-Processing Output

# %%
## Input data path
import pathlib

REPO_ROOT = pathlib.Path(".")

DATA_PATH = REPO_ROOT / "examples" / "example_data"
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Data path {DATA_PATH} does not exist. Please create the example_data directory and add the required data files."
    )
PREPROCESS_OUTPUT_PATH = DATA_PATH / "preprocessed_data"
if not PREPROCESS_OUTPUT_PATH.exists():
    raise FileNotFoundError(
        f"Preprocessed data path {PREPROCESS_OUTPUT_PATH} does not exist."
    )

PREPROCESS_SC_AUGMENTED_OUTPUT_PATH = PREPROCESS_OUTPUT_PATH / "sc_augmented"
if not PREPROCESS_SC_AUGMENTED_OUTPUT_PATH.exists():
    raise FileNotFoundError(
        f"Preprocessed single-cell augmented data path {PREPROCESS_SC_AUGMENTED_OUTPUT_PATH} does not exist."
    )

CIBERSORTX_SIG_GENE_FILE = (
    DATA_PATH
    / "cibersort_liver"
    / "CIBERSORTx_Job20_all-liver_0_cybersort_sig_inferred_phenoclasses.CIBERSORTx_Job20_all-liver_0_cybersort_sig_inferred_refsample.bm.K999.txt"
)
if not CIBERSORTX_SIG_GENE_FILE.exists():
    raise FileNotFoundError(
        f"CIBERSORTx signature gene file {CIBERSORTX_SIG_GENE_FILE} does not exist. Please add the CIBERSORTx signature gene file to the example_data/cibersort_liver/ directory."
    )

PREPROCESS_BULK_FORMAT_PATH = PREPROCESS_OUTPUT_PATH / "bulk_formatted"
if not PREPROCESS_BULK_FORMAT_PATH.exists():
    raise FileNotFoundError('Please create the output directory "bulk_formatted" first')

# %% [markdown]
# ### Read cibersortx sig genes

# %%
cibersortx_sig_df = pd.read_csv(CIBERSORTX_SIG_GENE_FILE, sep="\t", header=0)
cibersortx_sig_genes = cibersortx_sig_df["NAME"].values.tolist()
print(f"{len(cibersortx_sig_genes)} signature genes from CIBERSORTx")

# %% [markdown]
# ### Read gene id pkl file for single cell and psuedobulk and find intersection

# %%
bulk_gene_file = PREPROCESS_BULK_FORMAT_PATH.glob("*_genes.pkl")
bulk_gene_file = list(bulk_gene_file)
if not len(bulk_gene_file) == 1:
    raise ValueError(f"Expected one bulk gene file, found {len(bulk_gene_file)}")
bulk_genes = pd.read_pickle(bulk_gene_file[0])
print(f"Number of bulk genes: {len(bulk_genes)}")

sc_gene_file = PREPROCESS_SC_AUGMENTED_OUTPUT_PATH.glob("*_genes.pkl")
sc_gene_file = list(sc_gene_file)
if not len(sc_gene_file) == 1:
    raise ValueError(f"Expected one sc gene file, found {len(sc_gene_file)}")
sc_genes = pd.read_pickle(sc_gene_file[0])
print(f"Number of single cell genes: {len(sc_genes)}")

# Intersect the genes
sc_bulk_intersection_genes = set(sc_genes) & set(bulk_genes)
sc_bulk_intersection_genes = list(sc_bulk_intersection_genes)
print(
    f"Number of genes in intersection of single cell and bulk: {len(sc_bulk_intersection_genes)}"
)

# drop cibersortx signature genes that are not in the sc bulk intersection
cibersortx_drop_genes = set(cibersortx_sig_genes) - set(sc_bulk_intersection_genes)
print(
    f"Number of CIBERSORTx signature genes dropped due to not in intersection: {len(cibersortx_drop_genes)}"
)
cibersortx_sig_genes = set(cibersortx_sig_genes).intersection(
    sc_bulk_intersection_genes
)
cibersortx_sig_genes = list(cibersortx_sig_genes)

venn3(
    [set(sc_genes), set(bulk_genes), set(cibersortx_sig_genes)],
    set_labels=["Single Cell", "Bulk", "CIBERSORTx"],
)

# %% [markdown]
# ### Preprocessed Bulk

# %%
formatted_bulk_files = list(PREPROCESS_BULK_FORMAT_PATH.glob("*.h5ad"))
if not len(formatted_bulk_files) > 0:
    raise ValueError(f"No formatted bulk files found in {PREPROCESS_BULK_FORMAT_PATH}")
processed_bulk = sc.read(formatted_bulk_files[0])

bulk_sample_ids = processed_bulk.obs[SAMPLE_ID_COL].unique()
n_bulk_samples = len(bulk_sample_ids)
print(f"Number of bulk samples: {n_bulk_samples}")

bulk_stims = processed_bulk.obs[STIM_COL].unique()
n_bulk_stims = len(bulk_stims)
print(f"Number of bulk stims: {n_bulk_stims}")

## Subset the bulk data to only include the intersection genes
processed_bulk = processed_bulk[
    :, processed_bulk.var[GENE_ID_COL].isin(sc_bulk_intersection_genes)
]

X_bulk_train = pd.DataFrame(
    processed_bulk.X,
    index=processed_bulk.obs[SAMPLE_ID_COL],
    columns=processed_bulk.var[GENE_ID_COL],
)

meta_bulk_train = processed_bulk.obs.loc[:, [SAMPLE_ID_COL, STIM_COL]]
meta_bulk_train["isTraining"] = "Train"
meta_bulk_train["cell_prop_type"] = "bulk"
meta_bulk_train["cell_type"] = "bulk"
meta_bulk_train["samp_type"] = "bulk"

# %% [markdown]
# ### Preprocessed single cell

# %%
pseudobulk_files = list(PREPROCESS_SC_AUGMENTED_OUTPUT_PATH.glob("*_splits.pkl"))
if not len(pseudobulk_files) > 0:
    raise ValueError(
        f"No pseudobulk files found in {PREPROCESS_SC_AUGMENTED_OUTPUT_PATH}"
    )

pattern = re.compile(
    r"^(.*)_((?:Train)|(?:Test))_((?:meta)|(?:prop)|(?:pseudo))_splits\.pkl$"
)

# Temporary dictionary to group by sample_id
temp_dict = defaultdict(lambda: defaultdict(dict))

for path in pseudobulk_files:
    file = path.name
    match = pattern.match(file)
    if match:
        sample_id, datasplit, datatype = match.groups()
        temp_dict[datasplit][sample_id][datatype] = path

# Organizing the sorted results
grouped_files = {
    "Train": {"meta": [], "prop": [], "pseudo": []},
    "Test": {"meta": [], "prop": [], "pseudo": []},
}

# Sort within each datasplit by sample_id and organize the lists
for datasplit in ["Train", "Test"]:
    sorted_samples = sorted(temp_dict[datasplit].keys())  # Sort by sample_id
    for sample_id in sorted_samples:
        for datatype in ["meta", "prop", "pseudo"]:
            print(f"Found {datatype} file for {datasplit} sample {sample_id}")
            if datatype in temp_dict[datasplit][sample_id]:
                grouped_files[datasplit][datatype].append(
                    temp_dict[datasplit][sample_id][datatype]
                )

# Train datasplit
meta_sc_train = pd.concat(
    [pd.read_pickle(file) for file in grouped_files["Train"]["meta"]]
)
meta_sc_train["isTraining"] = "Train"
Y_sc_train = pd.concat(
    [pd.read_pickle(file) for file in grouped_files["Train"]["prop"]]
)
X_sc_train = pd.concat(
    [
        pd.read_pickle(file).loc[:, sc_bulk_intersection_genes]
        for file in grouped_files["Train"]["pseudo"]
    ]
)

_n_samples = meta_sc_train[SAMPLE_ID_COL].nunique()
print(f"Number of train single cell samples: {_n_samples}")

_n_stims = meta_sc_train[STIM_COL].nunique()
print(f"Number of train single cell stims: {_n_stims}")

# Test datasplit
meta_sc_test = pd.concat(
    [pd.read_pickle(file) for file in grouped_files["Test"]["meta"]]
)
meta_sc_test["isTraining"] = "Test"
Y_sc_test = pd.concat([pd.read_pickle(file) for file in grouped_files["Test"]["prop"]])
X_sc_test = pd.concat(
    [
        pd.read_pickle(file).loc[:, sc_bulk_intersection_genes]
        for file in grouped_files["Test"]["pseudo"]
    ]
)

_n_samples = meta_sc_test[SAMPLE_ID_COL].nunique()
print(f"Number of test single cell samples: {_n_samples}")
_n_stims = meta_sc_test[STIM_COL].nunique()
print(f"Number of test single cell stims: {_n_stims}")

# %% [markdown]
# ### Concatenate X, Y, Perform Gene Feature Selection

# %%
# Generate place holder for bulk proportion (not used)
Y_bulk_dummy = pd.DataFrame(
    np.zeros((X_bulk_train.shape[0], Y_sc_train.shape[1])), columns=Y_sc_train.columns
)

X_concat = pd.concat([X_bulk_train, X_sc_train, X_sc_test])
Y_concat = pd.concat([Y_bulk_dummy, Y_sc_train, Y_sc_test])
meta_concat = pd.concat([meta_bulk_train, meta_sc_train, meta_sc_test])

# save gene and cell type names
X_gene_names = X_concat.columns.to_list()
Y_cell_type_names = Y_concat.columns.to_list()

## get the top variable genes
X_colmean = X_concat.values.mean(axis=0)  # mean across samples
X_colvar = X_concat.values.var(axis=0)  # variance across samples
# coefficient of variation which is the var to mean ratio
X_CoV = np.array(np.divide(X_colvar, X_colmean + 0.001))

# need to get the genes such that
# the union of the highly variable and the
# CIBERSORTx genes are 7000 total
num_genes_found = False

gene_df = pd.DataFrame(X_gene_names, columns=["gene"])

# start with top 7000 genes by CoV
initial_count = 7000
while not num_genes_found:
    # retrieve the top initial_count genes by CoV
    idx_top = np.argpartition(X_CoV, -initial_count)[-initial_count:]
    # get gene names from idx_top
    top_gene_df = gene_df.iloc[idx_top]

    # produce the union of the top genes and the cibersort genes
    CoV_only = np.union1d(top_gene_df, cibersortx_sig_genes)

    # check if the union is 7000
    if len(CoV_only) == 7000:
        num_genes_found = True
    else:
        # if not decrement the top CoV gene to retrieve by 1
        # in the next iteration, there will be two possibilities
        # 1. the union will have one less gene due to the removed gene being only in the CoV genes
        # 2. the union will have the same number of genes as this iteration due to the removed
        #    gene being also present in CIBERSORTx genes
        # keep running this loop until the union has 7000 genes
        initial_count = initial_count - 1

idx_top = np.argpartition(X_CoV, -initial_count)[
    -initial_count:
]  # num_genes to get 7000
gene_df = gene_df.iloc[idx_top]
venn2(
    [set(gene_df.values.flatten()), set(cibersortx_sig_genes)],
    set_labels=("Top CoV Genes", "Cibersort Genes"),
)
plt.show()

feature_select_genes = np.union1d(gene_df, cibersortx_sig_genes)

# to numpy matrix
X = X_concat.loc[:, feature_select_genes].to_numpy()
print(X.shape)

Y = Y_concat.to_numpy()

# %% [markdown]
# ### Normalize Expression

# %%
## library size normalization
expected_libsizes = 1e6

libsizes = np.sum(X, axis=1, keepdims=True)
libsizes[libsizes == 0] = 1
X_norm = X / libsizes * expected_libsizes

## normalize within sample
clip_upper = np.quantile(X_norm, 0.9)
X_full = np.clip(X_norm, 0, clip_upper)
scaler = MinMaxScaler()
scaler.fit(X_full)

# now normalize with the scaler trained on the
# training data
X_full = np.clip(X_full, 0, clip_upper)
X_full = scaler.transform(X_full)

# %% [markdown]
# ### Construct Data container

# %%
train_data = BuDDINData(
    X=X_full,
    y=Y,
    meta=meta_concat,
    gene_names=feature_select_genes,
    cell_type_names=Y_cell_type_names,
    split_column="isTraining",
    encode_fields=[  # tell the data class to one-hot encode these fields
        SAMPLE_ID_COL,
        STIM_COL,
        TECH_COL,
    ],
)

# %% [markdown]
# To access data:

# %%
(_X, _y, _label, _stim, _samp_type, _meta) = train_data.get()
print(f"X shape: {_X.shape}")
print(f"y shape: {_y.shape}")
print(f"label shape: {_label.shape}")
print(f"stim shape: {_stim.shape}")
print(f"samp_type shape: {_samp_type.shape}")
print(f"meta shape: {_meta.shape}")

# %% [markdown]
# To access select modalities:

# %%
_X = train_data.get(["X"])
print(f"X shape: {_X.shape}")

# %% [markdown]
# To select for specific data

# %%
_X, _meta = train_data.reset_query().query(isTraining="Train").get(("X", "meta"))
print(f"X shape: {_X.shape}")
_meta.head()

# %%
_X, _meta = (
    train_data.reset_query().query(isTraining="Train", stim="female").get(("X", "meta"))
)
print(f"X shape: {_X.shape}")
_meta.head()

# %% [markdown]
# To up/down-sample:

# %%
_X, _meta = (
    train_data.reset_query()
    .query(isTraining="Train", stim="male")
    .get(("X", "meta"), n_samples=100, replace=True, random_state=42)
)
print(f"X shape: {_X.shape}")
_meta.head()

# %% [markdown]
# ### Visualize Data

# %% [markdown]
# Plot everything

# %%
train_data.reset_query().plot(
    color_by=[
        SAMPLE_ID_COL,
        STIM_COL,
        TECH_COL,
        train_data.split_column,
        train_data.ct_column,
    ],
    panel_width=8,
    show_plot=True,
)

# %% [markdown]
# Plot only the train split

# %%
train_data.reset_query().query(isTraining="Train").plot(
    color_by=[
        SAMPLE_ID_COL,
        STIM_COL,
        TECH_COL,
        train_data.split_column,
        train_data.ct_column,
    ],
    panel_width=8,
    show_plot=True,
)

# %% [markdown]
# ### Save the data class for later use

# %%
_ = joblib.dump(train_data.reset_query(), DATA_PATH / "buddi4_data.pkl")

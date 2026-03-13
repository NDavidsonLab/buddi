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
# # This notebook formats bulk RNA-seq dataset

# %%
import gc
import pathlib
import gzip
import pickle

import pandas as pd
import numpy as np
import anndata as ad

# %% [markdown]
# ## Preprocessing Parameters

# %%
SAMPLE_ID_COL = "sample_id"
SAMPLE_COL_SOURCE = "source name"
STIM_COL = "stim"
STIM_COL_SOURCE = "characteristics: sex"


def get_stim_id(in_str):
    out_str = "female"
    if in_str == "m":
        out_str = "male"

    return out_str


GENE_ID_COL = "gene_ids"

DATASPLIT_SEED = 42

# %% [markdown]
# ## Retrieve Path to Processed Bulk RNA-seq Data and relevant Metadata

# %%
## Input data path
DATA_PATH = pathlib.Path(".").absolute() / "examples" / "example_data"
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Data path {DATA_PATH} does not exist.")
BULK_DATA_FILE = DATA_PATH / "bulk_data" / "GSE132040_190214.csv.gz"
if not BULK_DATA_FILE.exists():
    raise FileNotFoundError(
        f"Bulk data file {BULK_DATA_FILE} does not exist. Please add the bulk data file to the example_data/bulk_data/ directory."
    )
BULK_METADATA_FILE = DATA_PATH / "bulk_data" / "GSE132040_MACA_Bulk_metadata.csv"
if not BULK_METADATA_FILE.exists():
    raise FileNotFoundError(
        f"Bulk metadata file {BULK_METADATA_FILE} does not exist. Please add the bulk metadata file to the example_data/bulk_data/ directory."
    )

## Output path
PREPROCESS_OUTPUT_PATH = (
    pathlib.Path(".").absolute() / "examples" / "example_data" / "preprocessed_data"
)
if not PREPROCESS_OUTPUT_PATH.exists():
    raise FileNotFoundError(
        'Please create the output directory "preprocessed_data" first'
    )
PREPROCESS_BULK_FORMAT_PATH = PREPROCESS_OUTPUT_PATH / "bulk_formatted"
PREPROCESS_BULK_FORMAT_PATH.mkdir(parents=True, exist_ok=True)
PREPROCESS_BULK_FORMAT_FILE = (
    PREPROCESS_BULK_FORMAT_PATH / "GSE132040_190214_bulk_formatted.h5ad"
)
PREPROCESS_BULK_FORMAT_GENE_FILE = (
    PREPROCESS_BULK_FORMAT_PATH / "GSE132040_190214_bulk_genes.pkl"
)

# %% [markdown]
# ## Preprocessing of Bulk Data
# ### Assemble csv expression and csv metadata to an adata object

# %%
# Load the metadata file
metadata = pd.read_csv(BULK_METADATA_FILE, index_col=0)
metadata.index = metadata.index.astype(str) + ".gencode.vM19"

# Load the bulk data file as an AnnData object
with gzip.open(BULK_DATA_FILE, "rt") as f:
    X = pd.read_csv(f, index_col=0)
    X = X.T
    metadata = metadata.loc[X.index]  # order

# Assemble anndata object
adata = ad.AnnData(X, obs=metadata)

del X
del metadata
gc.collect()

# %% [markdown]
# ### Format metadata

# %%
# remove non-gene IDs
gene_idx = np.where(np.logical_not(adata.var_names.str.startswith("__")))[0]
adata = adata[:, gene_idx]

# format the tissue
adata.obs["tissue"] = [x.split("_")[0] for x in adata.obs["source name"]]

# subset to post-pubescent liver
adata = adata[np.where(adata.obs["tissue"] == "Liver")]
adata = adata[np.where(adata.obs["characteristics: age"] != "1")]

# %% [markdown]
# ### format for BuDDI and write

# %%
adata.obs[SAMPLE_ID_COL] = adata.obs[SAMPLE_COL_SOURCE]
adata.obs[STIM_COL] = [get_stim_id(str(x)) for x in adata.obs[STIM_COL_SOURCE].tolist()]
adata.var[GENE_ID_COL] = adata.var.index.tolist()

# %% [markdown]
# ### Sample vs Stim Contingency Table

# %%
ct = pd.crosstab(adata.obs[SAMPLE_ID_COL], adata.obs[STIM_COL])
with pd.option_context(
    "display.max_rows",
    None,
    "display.max_columns",
    None,
    "display.width",
    None,
    "display.max_colwidth",
    None,
):
    print(ct)

# %%
del adata.raw
adata.write(PREPROCESS_BULK_FORMAT_FILE)
pickle.dump(adata.var[GENE_ID_COL], open(PREPROCESS_BULK_FORMAT_GENE_FILE, "wb"))

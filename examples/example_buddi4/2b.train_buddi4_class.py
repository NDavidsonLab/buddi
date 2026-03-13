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
# # This notebook demonstrates how to train class version of buddi4 model from buddi_v2

# %%
import os
import joblib

import numpy as np
import pandas as pd

"""Remove line below to use GPU"""
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # This LINE disables GPU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import CategoricalCrossentropy, MeanAbsoluteError

# %%
SAMPLE_ID_COL = "sample_id"
STIM_COL = "stim"
GENE_ID_COL = "gene_ids"
TECH_COL = "samp_type"

# %% [markdown]
# ## Add buddi_v2 software to path

# %%
from buddi_v2.dataset.buddi4_dataset import (
    get_supervised_dataset,
    get_unsupervised_dataset,
)
from buddi_v2.models.components.losses import kl_loss
from buddi_v2.models.buddi4_class import BuDDI4
from buddi_v2.models.fit import fit_buddi
from buddi_v2.plotting.plot_loss import plot_loss
from buddi_v2.plotting.plot_latent_space import plot_latent_spaces_buddi

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

# %% [markdown]
# ### Load train data

# %%
train_data = joblib.load(DATA_PATH / "buddi4_data.pkl")
train_data.reset_query()
train_data

# %% [markdown]
# ### Supervised vs Unsupervised split & Dataset Generation

# %%
(X_kp, y_kp, sample_id_kp, stim_kp, samp_type_kp, meta_kp) = (
    train_data.reset_query()
    .query(isTraining="Train", samp_type="sc_ref")
    .get(("X", "y", SAMPLE_ID_COL, STIM_COL, TECH_COL, "meta"))
)

(X_unkp, y_unkp, sample_id_unkp, stim_unkp, samp_type_unkp, meta_unkp) = (
    train_data.reset_query()
    .query(isTraining="Train", samp_type="bulk")
    .get(("X", "y", SAMPLE_ID_COL, STIM_COL, TECH_COL, "meta"))
)

# %% [markdown]
# Generating Tensorflow `dataset` objects for BuDDI4 training (**note** `ds_sup` and `ds_unsup` do not work with plotting or model inference out of the box), use raw data (e.g. `X_kp`, `X_unkp`) instead for these purposes.

# %%
ds_sup = get_supervised_dataset(
    X_known_prop=X_kp,
    Y_known_prop=y_kp,
    label_known_prop=sample_id_kp,
    stim_known_prop=stim_kp,
    samp_type_known_prop=samp_type_kp,
)
print(f"Number of entries in supervised dataset: {ds_sup.cardinality().numpy()}")
ds_sup_batch_x, ds_sup_batch_y = next(iter(ds_sup))

ds_unsup = get_unsupervised_dataset(
    X_unknown_prop=X_unkp,
    label_unknown_prop=sample_id_unkp,
    stim_unknown_prop=stim_unkp,
    samp_type_unknown_prop=samp_type_unkp,
)
print(f"Number of entries in unsupervised dataset: {ds_unsup.cardinality().numpy()}")

print("\nTake a peek at the dataset generator return:\n")

ds_sup_batch_input, ds_sup_batch_target = next(iter(ds_sup.batch(16).take(1)))
ds_sup_batch_x, ds_sup_batch_y = ds_sup_batch_input
print(f"Supervised batch x shape: {ds_sup_batch_x.shape}")
print(f"Supervised batch y shape: {ds_sup_batch_y.shape}")
(
    ds_sup_batch_target_x,
    _,
    _,
    _,
    _,
    ds_sup_batch_label,
    ds_sup_batch_stim,
    ds_sup_batch_samp_type,
    ds_sup_batch_target_y,
) = ds_sup_batch_target
print(f"Supervised batch target x shape: {ds_sup_batch_target_x.shape}")
print(f"Supervised batch target y shape: {ds_sup_batch_target_y.shape}")
print(f"Supervised batch label shape: {ds_sup_batch_label.shape}")
print(f"Supervised batch stim shape: {ds_sup_batch_stim.shape}")
print(f"Supervised batch samp_type shape: {ds_sup_batch_samp_type.shape}")

print("\n")

ds_unsup_batch_input, ds_unsup_batch_target = next(iter(ds_unsup.batch(16).take(1)))
(ds_unsup_batch_x,) = ds_unsup_batch_input
print(f"Unsupervised batch x shape: {ds_unsup_batch_x.shape}")
(
    ds_unsup_batch_target_x,
    _,
    _,
    _,
    _,
    ds_unsup_batch_label,
    ds_unsup_batch_stim,
    ds_unsup_batch_samp_type,
    _,
) = ds_unsup_batch_target
print(f"Unsupervised batch target x shape: {ds_unsup_batch_target_x.shape}")
print(f"Unsupervised batch label shape: {ds_unsup_batch_label.shape}")
print(f"Unsupervised batch stim shape: {ds_unsup_batch_stim.shape}")
print(f"Unsupervised batch samp_type shape: {ds_unsup_batch_samp_type.shape}")

# %% [markdown]
# ## BuDDI Training

# %% [markdown]
# ### Instantiate Model

# %%
train_data._encode_fields

# %%
n_x = train_data.shape["X"]
n_y = train_data.shape["y"]
n_sample_ids = train_data.shape["n_sample_id"]
n_stims = train_data.shape["n_stim"]
n_samp_types = train_data.shape["n_samp_type"]

# %%
obj = BuDDI4(
    n_x=n_x,
    n_y=n_y,
    n_sample_ids=n_sample_ids,
    n_stims=n_stims,
    n_samp_types=n_samp_types,
    z_dim=64,
    encoder_hidden_dim=512,
    decoder_hidden_dim=512,
    activation="relu",
    output_activation="sigmoid",
)

# %% [markdown]
# ### Configure Losses

# %%
ALPHA = 100.0
ALPHA_X = 1.0

BETA = 100.0
BETA_SAMPLE = 10.0
BETA_SLACK = 0.1

obj.set_reconstruction_loss(
    fn=MeanAbsoluteError(reduction="sum"),
    weight=ALPHA_X,
)
obj.set_encoder_loss(
    branch_name=[
        SAMPLE_ID_COL,
    ],
    fn=kl_loss,
    weight=BETA_SAMPLE,
)
obj.set_encoder_loss(
    branch_name=[
        STIM_COL,
        TECH_COL,
    ],
    fn=kl_loss,
    weight=BETA,
)
obj.set_encoder_loss(
    branch_name="slack",
    fn=kl_loss,
    weight=BETA_SLACK,
)
obj.set_predictor_loss(
    branch_name=[
        SAMPLE_ID_COL,
        STIM_COL,
        TECH_COL,
    ],
    fn=CategoricalCrossentropy(reduction="sum"),
    weight=ALPHA,
)
obj.set_prop_estimator_loss(
    fn=CategoricalCrossentropy(reduction="sum"),
    weight=ALPHA,
)

obj.print_loss_table()

# %% [markdown]
# ### Compile

# %%
obj.compile(optimizer=Adam(learning_rate=0.0005))

# %% [markdown]
# ### Visualize connection

# %%
plot_model(
    obj.sup_model,
    show_shapes=True,
    show_layer_names=True,
)

# %%
plot_model(
    obj.unsup_model,
    show_shapes=True,
    show_layer_names=True,
)

# %%
# !rm model.png

# %%
ds_test_sup = None
if train_data.reset_query().query(isTraining="Test", samp_type="sc_ref").__len__() > 0:
    (test_X_kp, test_y_kp, test_sample_id_kp, test_stim_kp, test_samp_type_kp) = (
        train_data.get(("X", "y", SAMPLE_ID_COL, STIM_COL, TECH_COL))
    )

    ds_test_sup = get_supervised_dataset(
        X_known_prop=test_X_kp,
        Y_known_prop=test_y_kp,
        label_known_prop=test_sample_id_kp,
        stim_known_prop=test_stim_kp,
        samp_type_known_prop=test_samp_type_kp,
    )

ds_test_unsup = None
if train_data.reset_query().query(isTraining="Test", samp_type="bulk").__len__() > 0:
    (
        test_X_unkp,
        test_y_unkp,
        test_sample_id_unkp,
        test_stim_unkp,
        test_samp_type_unkp,
    ) = train_data.get(("X", "y", SAMPLE_ID_COL, STIM_COL, TECH_COL))

    ds_test_unsup = get_unsupervised_dataset(
        X_unknown_prop=test_X_unkp,
        label_unknown_prop=test_sample_id_unkp,
        stim_unknown_prop=test_stim_unkp,
        samp_type_unknown_prop=test_samp_type_unkp,
    )

# %% [markdown]
# ### Train Model

# %%
all_loss_df = fit_buddi(
    supervised_model=obj.sup_model,
    unsupervised_model=obj.unsup_model,
    dataset_supervised=ds_sup,
    dataset_unsupervised=ds_unsup,
    dataset_test_supervised=ds_test_sup,
    dataset_test_unsupervised=ds_test_unsup,
    epochs=10,
    batch_size=16,
    shuffle_every_epoch=True,
    prefetch=False,
)

# %%
all_loss_df.head()

# %% [markdown]
# ### Examine Reconstruction
#
# **Optional**: For seeded deterministic behavior, do:
# ```python
# obj.set_reparam_deterministic(deterministic=True, seed=42)
# ```
# To go back to stochastic, simply do:
# ```python
# obj.set_reparam_deterministic(deterministic=False)
# ```

# %%
(
    x_hat,
    z_param_label,
    z_param_samp_type,
    z_param_stim,
    z_param_slack,
    pred_label,
    pred_stim,
    pred_samp_type,
    y_hat,
) = obj.unsup_model(ds_unsup_batch_input)
print(f"Predicted x shape: {x_hat.shape}")
print(f"Predicted z_param_label shape: {z_param_label.shape}")
print(f"Predicted z_param_stim shape: {z_param_stim.shape}")
print(f"Predicted z_param_samp_type shape: {z_param_samp_type.shape}")
print(f"Predicted z_param_slack shape: {z_param_slack.shape}")
print(f"Predicted pred_label shape: {pred_label.shape}")
print(f"Predicted pred_stim shape: {pred_stim.shape}")
print(f"Predicted pred_samp_type shape: {pred_samp_type.shape}")
print(f"Predicted y_hat shape: {y_hat.shape}")

# %%
# Down-sample the sc_ref and bulk data for visualization
(X_kp_tmp, Y_kp_tmp, meta_kp_tmp) = (
    train_data.reset_query()
    .query(isTraining="Train", samp_type="sc_ref")
    .get(("X", "y", "meta"), n_samples=500, replace=True, random_state=42)
)

(X_unkp_tmp, Y_unkp_tmp, meta_unkp_tmp) = (
    train_data.reset_query()
    .query(isTraining="Train", samp_type="bulk")
    .get(("X", "y", "meta"), n_samples=500, replace=True, random_state=42)
)

cell_types = train_data.cell_type_names
X_tmp = np.concatenate((X_kp_tmp, X_unkp_tmp), axis=0)
Y_tmp = np.concatenate((Y_kp_tmp, Y_unkp_tmp), axis=0)
meta_tmp = pd.concat((meta_kp_tmp, meta_unkp_tmp), axis=0)


idx_sc_prop = np.where(meta_tmp.cell_prop_type == "single_celltype")[0]
cell_type_labels = [cell_types[i] for i in np.argmax(Y_tmp[idx_sc_prop, :], axis=1)]
cell_type_col = meta_tmp.loc[:, "cell_prop_type"].values.copy()
cell_type_col[idx_sc_prop] = cell_type_labels
meta_tmp.loc[:, "cell_type"] = cell_type_col

_ = plot_latent_spaces_buddi(
    obj,
    X_tmp=X_tmp,
    meta_tmp=meta_tmp,
    type="PCA",
    panel_width=5,
    show_plot=True,
    save_path=None,
)

# %%
# Down-sample the sc_ref and bulk data and don't filter by split
(X_kp_tmp, Y_kp_tmp, meta_kp_tmp) = (
    train_data.reset_query()
    .query(samp_type="sc_ref")
    .get(("X", "y", "meta"), n_samples=500, replace=True, random_state=42)
)

(X_unkp_tmp, Y_unkp_tmp, meta_unkp_tmp) = (
    train_data.reset_query()
    .query(samp_type="bulk")
    .get(("X", "y", "meta"), n_samples=500, replace=True, random_state=42)
)

cell_types = train_data.cell_type_names
X_tmp = np.concatenate((X_kp_tmp, X_unkp_tmp), axis=0)
Y_tmp = np.concatenate((Y_kp_tmp, Y_unkp_tmp), axis=0)
meta_tmp = pd.concat((meta_kp_tmp, meta_unkp_tmp), axis=0)


idx_sc_prop = np.where(meta_tmp.cell_prop_type == "single_celltype")[0]
cell_type_labels = [cell_types[i] for i in np.argmax(Y_tmp[idx_sc_prop, :], axis=1)]
cell_type_col = meta_tmp.loc[:, "cell_prop_type"].values.copy()
cell_type_col[idx_sc_prop] = cell_type_labels
meta_tmp.loc[:, "cell_type"] = cell_type_col

_ = plot_latent_spaces_buddi(
    obj,
    X_tmp=X_tmp,
    meta_tmp=meta_tmp,
    type="PCA",
    panel_width=5,
    show_plot=True,
    save_path=None,
)

# %%
plot_loss(all_loss_df, save_path=None, show_plot=True)

# %% [markdown]
# ### Saving BuDDI4 object

# %%
obj.save(directory="temp")

# %% [markdown]
# Saved model structure looks like this

# %%
print(os.listdir("temp"))

# %%
# !rm -rf temp

# %% [markdown]
# ### To load saved object and weights, do

# %%
# loaded_obj = BuDDI4.load(
#     directory='temp'
# )

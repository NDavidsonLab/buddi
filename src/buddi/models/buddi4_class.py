from typing import Callable, List, Optional, Union

import tensorflow as tf

from .buddi_abstract_class import BuDDIAbstract

ActivationFn = Union[str, Callable[[tf.Tensor], tf.Tensor]]


class BuDDI4(BuDDIAbstract):
    """
    BuDDI4 class for building and training the BuDDI4 model.
    """

    def __init__(
        self,
        n_x: int,
        n_y: int,
        z_dim: int = 64,
        encoder_hidden_dim: int = 512,
        decoder_hidden_dim: int = 512,
        activation: ActivationFn = "relu",
        output_activation: ActivationFn = "relu",
        encoder_branch_names: Optional[List[str]] = None,
        slack_branch_name: Optional[str] = None,
        **kwargs,  # to collect the n_{branch_name} arguments
    ):
        """
        BuDDI4 model class constructor.

        :param n_x: Number of input features
        :param n_y: Number of output features
        :param z_dim: Latent space dimension
        :param encoder_hidden_dim: Hidden dimension of the encoder
        :param decoder_hidden_dim: Hidden dimension of the decoder
        :param activation: Activation function for the encoder and decoder
        :param output_activation: Output activation function for the decoder
        :param encoder_branch_names: List of encoder branch names
        :param slack_branch_name: Name of the slack branch
        :param kwargs: Meant for additional dimension arguments for the encoder branches
        """
        super().__init__(
            n_x=n_x,
            n_y=n_y,
            z_dim=z_dim,
            encoder_hidden_dim=encoder_hidden_dim,
            decoder_hidden_dim=decoder_hidden_dim,
            activation=activation,
            output_activation=output_activation,
            encoder_branch_names=encoder_branch_names,
            slack_branch_name=slack_branch_name,
            **kwargs,
        )

    # ─── Checker functions ───────────────────────────────────────────
    def default_encoder_branch_names(self):
        return ["sample_id", "stim", "samp_type"]

    def check_encoder_branch_names(self, encoder_branch_names: List[str]):
        """
        Check if the encoder branch names are valid.
        Subclasses should modify this method for rigorous input checks

        :param encoder_branch_names: List of encoder branch names
        """
        if len(encoder_branch_names) != 3:
            raise ValueError(
                f"Expected 3 encoder branch names, got {len(encoder_branch_names)}"
            )
        elif not all(isinstance(name, str) for name in encoder_branch_names):
            raise ValueError(
                f"Expected encoder branch names to be strings, got {type(encoder_branch_names[0])}"
            )

    def check_slack_branch_name(self, slack_branch_name: str):
        """
        Check if the slack branch name is valid.
        Subclasses should modify this method for rigorous input checks

        :param slack_branch_name: Name of the slack branch
        """
        if not isinstance(slack_branch_name, str):
            raise ValueError(
                f"Expected slack branch name to be a string, got {type(slack_branch_name)}"
            )
        if slack_branch_name in self.encoder_branch_names:
            raise ValueError(
                f"Slack branch name {slack_branch_name} cannot be in encoder branch names {self.encoder_branch_names}"
            )

    # ─── Fit ────────────────────────────────────────────────────

    def fit(self, **kwargs):
        raise NotImplementedError("fit() not implemented.")

from typing import Callable, List, Optional, Union

import tensorflow as tf

from .buddi_abstract_class import BuDDIAbstract

ActivationFn = Union[str, Callable[[tf.Tensor], tf.Tensor]]


class BuDDI(BuDDIAbstract):
    """
    Unified BuDDI model class supporting any number of compartments (encoder branches).

    This class replaces BuDDI3 and BuDDI4 by accepting a configurable number of
    encoder branches. The number of compartments equals the number of supervised
    encoder branches plus one slack branch.

    Examples:
        # 3-compartment model (equivalent to BuDDI3)
        model = BuDDI(
            n_x=1000,
            n_y=10,
            encoder_branch_names=['sample_id', 'samp_type'],
            n_sample_ids=50,
            n_samp_types=2
        )

        # 4-compartment model (equivalent to BuDDI4)
        model = BuDDI(
            n_x=1000,
            n_y=10,
            encoder_branch_names=['sample_id', 'stim', 'samp_type'],
            n_sample_ids=50,
            n_stims=3,
            n_samp_types=2
        )

        # 5-compartment model (custom)
        model = BuDDI(
            n_x=1000,
            n_y=10,
            encoder_branch_names=['sample_id', 'stim', 'samp_type', 'batch'],
            n_sample_ids=50,
            n_stims=3,
            n_samp_types=2,
            n_batchs=4
        )
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
        num_compartments: Optional[int] = None,
        **kwargs,  # to collect the n_{branch_name} arguments
    ):
        """
        BuDDI model class constructor supporting flexible number of compartments.

        :param n_x: Number of input features
        :param n_y: Number of output features
        :param z_dim: Latent space dimension
        :param encoder_hidden_dim: Hidden dimension of the encoder
        :param decoder_hidden_dim: Hidden dimension of the decoder
        :param activation: Activation function for the encoder and decoder
        :param output_activation: Output activation function for the decoder
        :param encoder_branch_names: List of encoder branch names. If None and
            num_compartments is provided, default names will be generated.
        :param slack_branch_name: Name of the slack branch
        :param num_compartments: Total number of compartments (including slack).
            If provided and encoder_branch_names is None, generic branch names
            will be created. Note: num_compartments = len(encoder_branch_names) + 1
        :param kwargs: Additional dimension arguments for the encoder branches
            (e.g., n_sample_ids=50, n_stims=3, etc.)
        """
        # Handle num_compartments parameter for convenience
        if encoder_branch_names is None and num_compartments is not None:
            if num_compartments < 2:
                raise ValueError(
                    f"num_compartments must be at least 2 (1 supervised + 1 slack), "
                    f"got {num_compartments}"
                )
            # Generate default branch names
            encoder_branch_names = [f"branch_{i}" for i in range(num_compartments - 1)]
            # Auto-populate kwargs with default dimensions if not provided
            for branch_name in encoder_branch_names:
                kwarg_key = f"n_{branch_name}s"
                if kwarg_key not in kwargs:
                    raise ValueError(
                        f"When using num_compartments, you must provide {kwarg_key} "
                        f"in kwargs (e.g., n_branch_{encoder_branch_names.index(branch_name)}s=...)"
                    )

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
        """
        Return default encoder branch names for a 3-compartment model (BuDDI3 style).
        Users should explicitly provide encoder_branch_names or num_compartments.
        """
        return ["sample_id", "samp_type"]

    def check_encoder_branch_names(self, encoder_branch_names: List[str]):
        """
        Check if the encoder branch names are valid.
        This flexible version accepts any number of branches >= 1.

        :param encoder_branch_names: List of encoder branch names
        """
        if len(encoder_branch_names) < 1:
            raise ValueError(
                f"Expected at least 1 encoder branch name, got {len(encoder_branch_names)}"
            )
        elif not all(isinstance(name, str) for name in encoder_branch_names):
            raise ValueError("Expected encoder branch names to be strings")
        # Check for duplicate names
        if len(encoder_branch_names) != len(set(encoder_branch_names)):
            raise ValueError(
                f"Encoder branch names must be unique, got duplicates in {encoder_branch_names}"
            )

    def check_slack_branch_name(self, slack_branch_name: str):
        """
        Check if the slack branch name is valid.

        :param slack_branch_name: Name of the slack branch
        """
        if not isinstance(slack_branch_name, str):
            raise ValueError(
                f"Expected slack branch name to be a string, got {type(slack_branch_name)}"
            )
        if slack_branch_name in self.encoder_branch_names:
            raise ValueError(
                f"Slack branch name '{slack_branch_name}' cannot be in encoder branch names"
            )

    # ─── Fit ────────────────────────────────────────────────────

    def fit(self, **kwargs):
        raise NotImplementedError("fit() not implemented.")

    @property
    def num_compartments(self) -> int:
        """
        Return the total number of compartments in the model.
        This equals the number of supervised encoder branches plus the slack branch.
        """
        return len(self.encoder_branch_names) + 1

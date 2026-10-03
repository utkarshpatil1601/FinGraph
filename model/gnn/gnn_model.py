"""
FinGraph - First-pass Heterogeneous GraphSAGE

Transaction fraud classification on the FinGraph HeteroData graph.

The model:
- projects each node type into a shared hidden space
- performs two HeteroConv + SAGEConv layers
- uses transaction/account relationships
- uses reverse message passing for Device/IP/Merchant/Session
- predicts fraud probability for Transaction nodes

This is intentionally a thin Phase 2 model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import HeteroConv, SAGEConv


# ============================================================
# NODE TYPES
# ============================================================

NODE_TYPES = [
    "transaction",
    "account",
    "device",
    "ip",
    "merchant",
    "session",
]


# ============================================================
# GRAPH RELATIONS USED BY THE MODEL
# ============================================================

BASE_RELATIONS = [
    ("account", "SENT", "transaction"),
    ("transaction", "SENT_BY", "account"),

    ("account", "RECEIVED", "transaction"),
    ("transaction", "RECEIVED_BY", "account"),

    ("account", "CONNECTED_TO", "account"),

    ("account", "USES_DEVICE", "device"),
    ("account", "USES_IP", "ip"),
    ("account", "USES_MERCHANT", "merchant"),
    ("account", "LOGIN_FROM", "session"),
]


# Reverse relations for static relationship nodes.
REVERSE_RELATIONS = [
    ("device", "REV_USES_DEVICE", "account"),
    ("ip", "REV_USES_IP", "account"),
    ("merchant", "REV_USES_MERCHANT", "account"),
    ("session", "REV_LOGIN_FROM", "account"),
]


MODEL_RELATIONS = (
    BASE_RELATIONS
    + REVERSE_RELATIONS
)


# ============================================================
# MODEL
# ============================================================

class FinGraphGraphSAGE(nn.Module):
    """
    Two-layer heterogeneous GraphSAGE.

    Output:
        One logit per transaction node.
    """

    def __init__(
        self,
        hidden_channels: int = 32,
        dropout: float = 0.30,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.dropout_rate = dropout

        # ----------------------------------------------------
        # Input projection
        #
        # LazyLinear determines each input feature dimension
        # on the first forward pass.
        # ----------------------------------------------------

        self.input_projection = nn.ModuleDict(
            {
                node_type: nn.LazyLinear(
                    hidden_channels
                )
                for node_type in NODE_TYPES
            }
        )

        # ----------------------------------------------------
        # First GraphSAGE layer
        # ----------------------------------------------------

        conv1 = {}

        for edge_type in MODEL_RELATIONS:

            conv1[edge_type] = SAGEConv(
                (
                    hidden_channels,
                    hidden_channels,
                ),
                hidden_channels,
                normalize=True,
            )

        self.conv1 = HeteroConv(
            conv1,
            aggr="sum",
        )

        # ----------------------------------------------------
        # Second GraphSAGE layer
        # ----------------------------------------------------

        conv2 = {}

        for edge_type in MODEL_RELATIONS:

            conv2[edge_type] = SAGEConv(
                (
                    hidden_channels,
                    hidden_channels,
                ),
                hidden_channels,
                normalize=True,
            )

        self.conv2 = HeteroConv(
            conv2,
            aggr="sum",
        )

        # ----------------------------------------------------
        # Classification head
        # ----------------------------------------------------

        self.classifier = nn.Sequential(
            nn.Linear(
                hidden_channels,
                hidden_channels,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_channels,
                1,
            ),
        )

        self.dropout = nn.Dropout(
            dropout
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x_dict: dict[str, Tensor],
        edge_index_dict: dict,
    ) -> Tensor:

        # ----------------------------------------------------
        # Project every node type to same dimension
        # ----------------------------------------------------

        x_dict = {
            node_type:
                self.input_projection[node_type](
                    x
                )
            for node_type, x in x_dict.items()
        }

        # ----------------------------------------------------
        # GraphSAGE layer 1
        # ----------------------------------------------------

        x_dict = self.conv1(
            x_dict,
            edge_index_dict,
        )

        x_dict = {
            node_type: torch.relu(
                x
            )
            for node_type, x in x_dict.items()
        }

        x_dict = {
            node_type: self.dropout(x)
            for node_type, x in x_dict.items()
        }

        # ----------------------------------------------------
        # GraphSAGE layer 2
        # ----------------------------------------------------

        x_dict = self.conv2(
            x_dict,
            edge_index_dict,
        )

        x_dict = {
            node_type: torch.relu(
                x
            )
            for node_type, x in x_dict.items()
        }

        # ----------------------------------------------------
        # Transaction classifier
        # ----------------------------------------------------

        transaction_embeddings = (
            x_dict["transaction"]
        )

        logits = self.classifier(
            transaction_embeddings
        )

        return logits.squeeze(-1)


# ============================================================
# ADD REVERSE STATIC RELATIONS
# ============================================================

def add_reverse_relationships(data):
    """
    Add reverse edges for:
        Account -> Device
        Account -> IP
        Account -> Merchant
        Account -> Session

    The stored graph is not modified on disk. This operates
    on the in-memory HeteroData object used by the GNN.
    """

    reverse_pairs = [
        (
            ("account", "USES_DEVICE", "device"),
            ("device", "REV_USES_DEVICE", "account"),
        ),
        (
            ("account", "USES_IP", "ip"),
            ("ip", "REV_USES_IP", "account"),
        ),
        (
            ("account", "USES_MERCHANT", "merchant"),
            ("merchant", "REV_USES_MERCHANT", "account"),
        ),
        (
            ("account", "LOGIN_FROM", "session"),
            ("session", "REV_LOGIN_FROM", "account"),
        ),
    ]

    for forward_edge, reverse_edge in reverse_pairs:

        if forward_edge not in data.edge_types:
            raise ValueError(
                f"Missing forward edge: {forward_edge}"
            )

        forward_index = data[
            forward_edge
        ].edge_index

        data[
            reverse_edge
        ].edge_index = forward_index.flip(
            dims=[0]
        )

    return data


# ============================================================
# MODEL RELATION VALIDATION
# ============================================================

def validate_model_relations(data):
    """
    Verify that all relations required by GraphSAGE exist.
    """

    # Original graph relations
    required_original = BASE_RELATIONS

    for edge_type in required_original:

        if edge_type not in data.edge_types:

            raise ValueError(
                f"Missing graph relation: {edge_type}"
            )

    # Reverse relations
    required_reverse = REVERSE_RELATIONS

    for edge_type in required_reverse:

        if edge_type not in data.edge_types:

            raise ValueError(
                f"Missing reverse graph relation: {edge_type}"
            )


# ============================================================
# CREATE MODEL
# ============================================================

def create_model(
    hidden_channels: int = 32,
    dropout: float = 0.30,
):
    """
    Convenience constructor.
    """

    return FinGraphGraphSAGE(
        hidden_channels=hidden_channels,
        dropout=dropout,
    )
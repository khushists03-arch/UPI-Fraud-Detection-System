from collections import defaultdict
from itertools import combinations
from typing import Dict, Iterable, Tuple

import pandas as pd
import torch


# These columns define relationships between transactions.
ENTITY_COLUMNS = [
    "user_id",
    "merchant_id",
    "device_id",
]


def build_transaction_graph(
    df: pd.DataFrame,
    entity_columns: Iterable[str] = ENTITY_COLUMNS,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Build a transaction-level graph for UPI fraud detection.

    Graph design:

        Transaction = Node

        Shared user_id      -> Edge
        Shared merchant_id  -> Edge
        Shared device_id    -> Edge

    If two transactions share more than one entity,
    their edge receives a higher weight.

    Returns
    -------
    edge_index:
        Tensor of shape [2, number_of_edges]

    edge_weight:
        Tensor containing the weight of every edge.

    stats:
        Dictionary containing graph information.
    """

    # Number of transaction nodes.
    num_nodes = len(df)

    # Store relationship strength between pairs of nodes.
    edge_weights = defaultdict(float)

    # Only use entity columns that actually exist.
    available_columns = [
        column
        for column in entity_columns
        if column in df.columns
    ]

    if not available_columns:

        raise ValueError(
            "No graph entity columns were found. "
            "Expected at least one of: "
            + ", ".join(ENTITY_COLUMNS)
        )

    # ---------------------------------------------------------
    # Build relationships
    # ---------------------------------------------------------

    for column in available_columns:

        # Group transactions according to the entity.
        groups = df.groupby(
            column,
            sort=False,
            dropna=False,
        ).groups

        # Each group contains transactions sharing
        # the same user / merchant / device.
        for _, indices in groups.items():

            indices = list(indices)

            # A group containing one transaction
            # cannot create an edge.
            if len(indices) < 2:
                continue

            # Create every pair inside the group.
            for i, j in combinations(indices, 2):

                if i == j:
                    continue

                # Keep ordering consistent.
                if i < j:
                    pair = (i, j)
                else:
                    pair = (j, i)

                # Increase relationship strength.
                edge_weights[pair] += 1.0

    # ---------------------------------------------------------
    # Convert relationships into undirected edges
    # ---------------------------------------------------------

    source_nodes = []
    destination_nodes = []
    weights = []

    for (i, j), weight in edge_weights.items():

        # i -> j
        source_nodes.append(i)
        destination_nodes.append(j)
        weights.append(weight)

        # j -> i
        source_nodes.append(j)
        destination_nodes.append(i)
        weights.append(weight)

    # ---------------------------------------------------------
    # Add self-loops
    # ---------------------------------------------------------

    # A self-loop allows each transaction to retain
    # information from its own node features.
    for i in range(num_nodes):

        source_nodes.append(i)
        destination_nodes.append(i)
        weights.append(1.0)

    # ---------------------------------------------------------
    # Convert to PyTorch tensors
    # ---------------------------------------------------------

    edge_index = torch.tensor(
        [
            source_nodes,
            destination_nodes,
        ],
        dtype=torch.long,
    )

    edge_weight = torch.tensor(
        weights,
        dtype=torch.float32,
    )

    # ---------------------------------------------------------
    # Graph statistics
    # ---------------------------------------------------------

    stats = {
        "num_nodes": int(num_nodes),

        "num_edges_including_self_loops": int(
            edge_index.shape[1]
        ),

        "num_unique_undirected_relationships": int(
            len(edge_weights)
        ),

        "entity_columns_used": available_columns,
    }

    return (
        edge_index,
        edge_weight,
        stats,
    )


def normalize_edge_weights(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """
    Perform symmetric GCN normalization.

    Formula:

        D^(-1/2) A D^(-1/2)

    This prevents nodes with many connections
    from dominating the message passing process.
    """

    source = edge_index[0]
    destination = edge_index[1]

    # Calculate degree for every node.
    degree = torch.zeros(
        num_nodes,
        dtype=edge_weight.dtype,
        device=edge_weight.device,
    )

    degree.index_add_(
        0,
        destination,
        edge_weight,
    )

    # D^(-1/2)
    degree_inv_sqrt = (
        degree
        .clamp_min(1e-12)
        .pow(-0.5)
    )

    # D^(-1/2) A D^(-1/2)
    normalized_weights = (
        edge_weight
        * degree_inv_sqrt[source]
        * degree_inv_sqrt[destination]
    )

    return normalized_weights


def find_related_nodes(
    reference_df: pd.DataFrame,
    transaction: pd.Series,
    entity_columns: Iterable[str] = ENTITY_COLUMNS,
):
    """
    Find historical transactions related to a new transaction.

    A historical transaction is considered related when it shares
    the same user, merchant, or device.
    """

    related_nodes = set()

    for column in entity_columns:

        if column not in reference_df.columns:
            continue

        if column not in transaction.index:
            continue

        value = transaction[column]

        if pd.isna(value):
            continue

        matches = reference_df.index[
            reference_df[column]
            .astype(str)
            .eq(str(value))
        ]

        related_nodes.update(
            int(index)
            for index in matches
        )

    return sorted(
        related_nodes
    )
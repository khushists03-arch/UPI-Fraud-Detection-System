"""
GNN package for the UPI Fraud Detection System.

This file imports the important GNN components
so they can be easily used from the package.
"""

# Import the GNN model
from .gnn_model import (
    GCNLayer,
    GCNClassifier,
    create_gnn_model
)

# Import functions used to build the graph
from .graph_builder import (
    build_transaction_graph,
    normalize_edge_weights,
    find_related_nodes
)

# Import the feature preprocessing class
from .preprocessing import (
    TransactionFeaturePreprocessor
)


# Important components of the GNN package
__all__ = [
    "GCNLayer",
    "GCNClassifier",
    "create_gnn_model",

    "build_transaction_graph",
    "normalize_edge_weights",
    "find_related_nodes",

    "TransactionFeaturePreprocessor",
]
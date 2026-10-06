"""
Graph Neural Network model for UPI fraud detection.

This module implements a lightweight Graph Convolutional Network (GCN)
without requiring PyTorch Geometric.

Each transaction is represented as a graph node.
The graph structure is created from relationships between transactions,
such as shared users, merchants, and devices.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """
    A single Graph Convolutional Network layer.

    The layer performs:

        H' = A_hat H W

    where:
        H      = input node features
        W      = learnable weight matrix
        A_hat  = normalized graph adjacency matrix

    The normalized edge weights are supplied by graph_builder.py.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.linear = nn.Linear(
            input_dim,
            output_dim,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform graph convolution.

        Parameters
        ----------
        x:
            Node feature matrix.

            Shape:
                [num_nodes, input_dim]

        edge_index:
            Graph connectivity.

            Shape:
                [2, num_edges]

            edge_index[0] contains source nodes.
            edge_index[1] contains destination nodes.

        edge_weight:
            Normalized weight for every graph edge.

            Shape:
                [num_edges]

        Returns
        -------
        torch.Tensor
            Updated node representations.

            Shape:
                [num_nodes, output_dim]
        """

        # First transform every node's features.
        transformed = self.linear(x)

        # Get source and destination nodes.
        source = edge_index[0]
        destination = edge_index[1]

        # Create an empty tensor for message aggregation.
        aggregated = torch.zeros_like(transformed)

        # Send messages from source nodes to destination nodes.
        messages = transformed[source] * edge_weight.unsqueeze(1)

        aggregated.index_add_(
            0,
            destination,
            messages,
        )

        return aggregated


class GCNClassifier(nn.Module):
    """
    Two-layer GCN classifier for transaction-level fraud detection.

    Architecture:

        Input Features
             |
             v
        GCN Layer 1
             |
             v
           ReLU
             |
             v
          Dropout
             |
             v
        GCN Layer 2
             |
             v
          ReLU
             |
             v
        Embeddings
             |
             v
        Linear Classifier
             |
             v
        Fraud / Genuine

    The model returns both:
        1. Classification logits
        2. Learned node embeddings

    The embeddings can later be used for:
        - fraud analysis
        - visualization
        - similarity analysis
        - downstream ML tasks
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be greater than zero.")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be greater than zero.")

        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout

        # First graph convolution.
        self.gcn1 = GCNLayer(
            input_dim=input_dim,
            output_dim=hidden_dim,
        )

        # Second graph convolution.
        self.gcn2 = GCNLayer(
            input_dim=hidden_dim,
            output_dim=hidden_dim,
        )

        # Final classifier.
        self.classifier = nn.Linear(
            hidden_dim,
            num_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the GCN.

        Parameters
        ----------
        x:
            Node feature matrix.

        edge_index:
            Graph edge indices.

        edge_weight:
            Normalized edge weights.

        Returns
        -------
        logits:
            Classification logits.

            Shape:
                [num_nodes, num_classes]

        embeddings:
            Learned graph/node representation.

            Shape:
                [num_nodes, hidden_dim]
        """

        # First GCN layer.
        x = self.gcn1(
            x,
            edge_index,
            edge_weight,
        )

        # Non-linear activation.
        x = F.relu(x)

        # Dropout during training.
        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        # Second GCN layer.
        embeddings = self.gcn2(
            x,
            edge_index,
            edge_weight,
        )

        # Non-linear activation.
        embeddings = F.relu(embeddings)

        # Classification layer.
        logits = self.classifier(embeddings)

        return logits, embeddings

    def predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return predicted class for every transaction.

        Returns
        -------
        torch.Tensor
            Predicted class indices.

            0 = genuine
            1 = fraud
        """

        self.eval()

        with torch.no_grad():
            logits, _ = self.forward(
                x,
                edge_index,
                edge_weight,
            )

            predictions = torch.argmax(
                logits,
                dim=1,
            )

        return predictions

    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return class probabilities for every transaction.

        Returns
        -------
        torch.Tensor
            Probability matrix.

            Column 0 = genuine probability
            Column 1 = fraud probability
        """

        self.eval()

        with torch.no_grad():
            logits, _ = self.forward(
                x,
                edge_index,
                edge_weight,
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

        return probabilities


def create_gnn_model(
    input_dim: int,
    hidden_dim: int = 64,
    num_classes: int = 2,
    dropout: float = 0.35,
) -> GCNClassifier:
    """
    Factory function for creating the GCN model.

    This keeps model creation consistent across training
    and prediction scripts.
    """

    return GCNClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
    )


__all__ = [
    "GCNLayer",
    "GCNClassifier",
    "create_gnn_model",
]
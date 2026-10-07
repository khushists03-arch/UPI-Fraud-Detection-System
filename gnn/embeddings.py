"""
Generate GNN embeddings for the UPI Fraud Detection System.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from gnn.preprocessing import TransactionFeaturePreprocessor
from gnn.graph_builder import (
    build_transaction_graph,
    normalize_edge_weights,
)
from gnn.gnn_model import GCNClassifier


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "fraud_dataset.csv"

ORIGINAL_DATA_PATH = (
    PROJECT_ROOT.parent
    / "UPI-Fraud-Detection-System"
    / "data"
    / "raw"
    / "fraud_dataset.csv"
)

MODEL_PATH = PROJECT_ROOT / "models" / "gnn_model.pt"

PREPROCESSOR_PATH = (
    PROJECT_ROOT / "models" / "gnn_preprocessor.joblib"
)

EMBEDDINGS_PATH = (
    PROJECT_ROOT / "models" / "gnn_embeddings.csv"
)


def get_dataset_path():
    """Find the fraud dataset."""

    if DATA_PATH.exists():
        return DATA_PATH

    if ORIGINAL_DATA_PATH.exists():
        return ORIGINAL_DATA_PATH

    raise FileNotFoundError(
        "fraud_dataset.csv was not found."
    )


def load_gnn_model(device):
    """Load the trained GNN model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "GNN model not found. "
            "Run: python -m gnn.train_gnn"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    model = GCNClassifier(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_classes=checkpoint["num_classes"],
        dropout=checkpoint["dropout"],
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model


def load_preprocessor():
    """Load the preprocessing pipeline."""

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            "GNN preprocessor not found. "
            "Run: python -m gnn.train_gnn"
        )

    return joblib.load(
        PREPROCESSOR_PATH
    )


def generate_embeddings(
    df,
    preprocessor,
    model,
    device,
):
    """Generate one embedding for each transaction."""

    # Convert transactions into model features.
    features_np = preprocessor.transform(df)

    features = torch.tensor(
        features_np,
        dtype=torch.float32,
        device=device,
    )

    # Build the transaction graph.
    edge_index, edge_weight, stats = (
        build_transaction_graph(df)
    )

    print(
        f"Graph nodes: {stats['num_nodes']}"
    )

    print(
        f"Graph edges: "
        f"{stats['num_edges_including_self_loops']}"
    )

    # Normalize graph weights.
    edge_weight = normalize_edge_weights(
        edge_index,
        edge_weight,
        len(df),
    )

    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    # Generate embeddings.
    with torch.no_grad():

        _, embeddings = model(
            features,
            edge_index,
            edge_weight,
        )

    return embeddings.cpu().numpy()


def save_embeddings(df, embeddings):
    """Save GNN embeddings to CSV."""

    embedding_columns = [
        f"embedding_{i}"
        for i in range(embeddings.shape[1])
    ]

    embedding_df = pd.DataFrame(
        embeddings,
        columns=embedding_columns,
    )

    # Keep transaction identifiers.
    output_df = pd.DataFrame()

    for column in [
        "transaction_id",
        "user_id",
        "merchant_id",
        "device_id",
    ]:
        if column in df.columns:
            output_df[column] = df[column].values

    output_df.insert(
        0,
        "gnn_node_index",
        np.arange(len(df)),
    )

    output_df = pd.concat(
        [output_df, embedding_df],
        axis=1,
    )

    # Keep fraud label only for analysis.
    if "is_fraud" in df.columns:
        output_df["is_fraud"] = df[
            "is_fraud"
        ].values

    EMBEDDINGS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_df.to_csv(
        EMBEDDINGS_PATH,
        index=False,
    )

    print(
        f"\nEmbeddings saved to:\n"
        f"{EMBEDDINGS_PATH}"
    )

    print(
        f"Rows: {len(output_df)}"
    )

    print(
        f"Embedding size: "
        f"{embeddings.shape[1]}"
    )


def main():

    print("=" * 60)
    print("GNN EMBEDDING GENERATION")
    print("=" * 60)

    # Use GPU if available.
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nDevice: {device}")

    # Load dataset.
    dataset_path = get_dataset_path()

    print(
        f"\nDataset:\n{dataset_path}"
    )

    df = pd.read_csv(
        dataset_path
    )

    print(
        f"Dataset shape: {df.shape}"
    )

    # Required graph columns.
    required_columns = [
        "user_id",
        "merchant_id",
        "device_id",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing columns: "
            + ", ".join(missing)
        )

    # Load trained GNN components.
    preprocessor = load_preprocessor()

    model = load_gnn_model(
        device
    )

    # Generate embeddings.
    embeddings = generate_embeddings(
        df,
        preprocessor,
        model,
        device,
    )

    # Save embeddings.
    save_embeddings(
        df,
        embeddings,
    )

    print(
        "\nGNN embedding generation complete."
    )


if __name__ == "__main__":
    main()
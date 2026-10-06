Project-specific pipeline:

    fraud_dataset.csv
            |
            v
    TransactionFeaturePreprocessor
            |
            v
    Transaction Features
            |
            v
    Transaction Graph
            |
            v
       Trained GCN
            |
            v
    64-Dimensional Embeddings
            |
            v
    models/gnn_embeddings.csv

Each row in the output represents one UPI transaction.

The embedding captures information learned from:
    - transaction features
    - shared users
    - shared merchants
    - shared devices
    - neighboring transactions
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from gnn.preprocessing import (
    TransactionFeaturePreprocessor,
)

from gnn.graph_builder import (
    build_transaction_graph,
    normalize_edge_weights,
)

from gnn.gnn_model import (
    GCNClassifier,
)

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


# Dataset inside the GNN branch.
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "fraud_dataset.csv"
)


# Fallback to the original project dataset.
ORIGINAL_PROJECT_DATA_PATH = (
    PROJECT_ROOT.parent
    / "UPI-Fraud-Detection-System"
    / "data"
    / "raw"
    / "fraud_dataset.csv"
)


# Trained GNN model.
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "gnn_model.pt"
)


# Saved preprocessing pipeline.
PREPROCESSOR_PATH = (
    PROJECT_ROOT
    / "models"
    / "gnn_preprocessor.joblib"
)


# Output embedding file.
EMBEDDINGS_PATH = (
    PROJECT_ROOT
    / "models"
    / "gnn_embeddings.csv"
)


# ============================================================
# DATASET LOCATION
# ============================================================

def get_dataset_path() -> Path:
    """
    Locate the fraud dataset.

    First checks the GNN branch.

    If it is not present there, checks the original
    UPI Fraud Detection project.
    """

    if DATA_PATH.exists():
        return DATA_PATH

    if ORIGINAL_PROJECT_DATA_PATH.exists():
        return ORIGINAL_PROJECT_DATA_PATH

    raise FileNotFoundError(
        "\nUPI fraud dataset was not found.\n\n"
        f"Checked:\n"
        f"1. {DATA_PATH}\n"
        f"2. {ORIGINAL_PROJECT_DATA_PATH}\n\n"
        "Expected file:\n"
        "fraud_dataset.csv"
    )


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_gnn_model(
    device: torch.device,
):
    """
    Load the trained GCN model from the checkpoint.

    The checkpoint created by train_gnn.py contains:
        - model_state_dict
        - input_dim
        - hidden_dim
        - num_classes
        - dropout
    """

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "\nTrained GNN model not found:\n"
            f"{MODEL_PATH}\n\n"
            "Train the GNN first using:\n"
            "python -m gnn.train_gnn"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    model = GCNClassifier(
        input_dim=checkpoint[
            "input_dim"
        ],
        hidden_dim=checkpoint[
            "hidden_dim"
        ],
        num_classes=checkpoint[
            "num_classes"
        ],
        dropout=checkpoint[
            "dropout"
        ],
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(device)

    model.eval()

    print(
        "\nGNN model loaded successfully."
    )

    print(
        f"Input features: "
        f"{checkpoint['input_dim']}"
    )

    print(
        f"Hidden dimension: "
        f"{checkpoint['hidden_dim']}"
    )

    print(
        f"Output classes: "
        f"{checkpoint['num_classes']}"
    )

    return model


# ============================================================
# LOAD PREPROCESSOR
# ============================================================

def load_preprocessor():
    """
    Load the preprocessing pipeline that was fitted
    during GNN training.
    """

    if not PREPROCESSOR_PATH.exists():

        raise FileNotFoundError(
            "\nGNN preprocessor not found:\n"
            f"{PREPROCESSOR_PATH}\n\n"
            "Train the GNN first using:\n"
            "python -m gnn.train_gnn"
        )

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    print(
        "\nGNN preprocessor loaded."
    )

    print(
        f"Input features: "
        f"{preprocessor.input_dim}"
    )

    return preprocessor


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

def generate_embeddings(
    df: pd.DataFrame,
    preprocessor: TransactionFeaturePreprocessor,
    model: GCNClassifier,
    device: torch.device,
) -> np.ndarray:
    """
    Generate one GNN embedding for every transaction.

    Parameters
    ----------
    df:
        UPI transaction dataframe.

    preprocessor:
        Fitted TransactionFeaturePreprocessor.

    model:
        Trained GCNClassifier.

    device:
        CPU or CUDA device.

    Returns
    -------
    np.ndarray
        Shape:

            number_of_transactions x embedding_dimension
    """

    print(
        "\nPreparing transaction features..."
    )

    features_np = (
        preprocessor.transform(df)
    )

    print(
        f"Feature matrix shape: "
        f"{features_np.shape}"
    )

    features = torch.tensor(
        features_np,
        dtype=torch.float32,
        device=device,
    )

    # --------------------------------------------------------
    # Build the same transaction graph used during training.
    #
    # Transactions are connected through:
    #   user_id
    #   merchant_id
    #   device_id
    # --------------------------------------------------------

    print(
        "\nBuilding transaction graph..."
    )

    (
        edge_index,
        edge_weight,
        graph_stats,
    ) = build_transaction_graph(
        df
    )

    print(
        f"Graph nodes: "
        f"{graph_stats['num_nodes']}"
    )

    print(
        f"Unique relationships: "
        f"{graph_stats['num_unique_undirected_relationships']}"
    )

    print(
        f"Graph edges: "
        f"{graph_stats['num_edges_including_self_loops']}"
    )

    # --------------------------------------------------------
    # Normalize graph weights.
    # --------------------------------------------------------

    normalized_edge_weight = (
        normalize_edge_weights(
            edge_index,
            edge_weight,
            num_nodes=len(df),
        )
    )

    edge_index = edge_index.to(
        device
    )

    normalized_edge_weight = (
        normalized_edge_weight.to(
            device
        )
    )

    # --------------------------------------------------------
    # Run GCN.
    # --------------------------------------------------------

    print(
        "\nGenerating GNN embeddings..."
    )

    model.eval()

    with torch.no_grad():

        _logits, embeddings = model(
            features,
            edge_index,
            normalized_edge_weight,
        )

    embeddings_np = (
        embeddings
        .cpu()
        .numpy()
    )

    print(
        f"Embedding matrix shape: "
        f"{embeddings_np.shape}"
    )

    return embeddings_np


# ============================================================
# SAVE EMBEDDINGS
# ============================================================

def save_embeddings(
    df: pd.DataFrame,
    embeddings: np.ndarray,
) -> None:
    """
    Save transaction embeddings to CSV.

    Output columns:

        gnn_node_index
        embedding_0
        embedding_1
        ...
        embedding_63
        is_fraud

    Original transaction identifiers are also retained
    when available.
    """

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    embedding_dimension = (
        embeddings.shape[1]
    )

    embedding_columns = [
        f"embedding_{index}"
        for index in range(
            embedding_dimension
        )
    ]

    embeddings_df = pd.DataFrame(
        embeddings,
        columns=embedding_columns,
    )

    # --------------------------------------------------------
    # Preserve transaction identifiers.
    # --------------------------------------------------------

    identifier_columns = [
        "transaction_id",
        "user_id",
        "merchant_id",
        "device_id",
    ]

    available_identifiers = [
        column
        for column in identifier_columns
        if column in df.columns
    ]

    output_df = pd.DataFrame()

    for column in available_identifiers:

        output_df[column] = (
            df[column].values
        )

    # --------------------------------------------------------
    # GNN node index.
    # --------------------------------------------------------

    output_df.insert(
        0,
        "gnn_node_index",
        np.arange(
            len(df)
        ),
    )

    # --------------------------------------------------------
    # Add embeddings.
    # --------------------------------------------------------

    output_df = pd.concat(
        [
            output_df,
            embeddings_df,
        ],
        axis=1,
    )

    # --------------------------------------------------------
    # Add target for analysis/evaluation.
    #
    # This target is NOT used to generate embeddings.
    # --------------------------------------------------------

    if "is_fraud" in df.columns:

        output_df[
            "is_fraud"
        ] = df[
            "is_fraud"
        ].values

    # --------------------------------------------------------
    # Save.
    # --------------------------------------------------------

    output_df.to_csv(
        EMBEDDINGS_PATH,
        index=False,
    )

    print(
        "\nEmbeddings saved successfully:"
    )

    print(
        EMBEDDINGS_PATH
    )

    print(
        f"\nRows: "
        f"{len(output_df)}"
    )

    print(
        f"Embedding dimensions: "
        f"{embedding_dimension}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Complete embedding generation pipeline.
    """

    print("=" * 70)

    print(
        "UPI FRAUD DETECTION - GNN EMBEDDING GENERATION"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset_path = (
        get_dataset_path()
    )

    print(
        f"\nDataset:"
        f"\n{dataset_path}"
    )

    df = pd.read_csv(
        dataset_path
    )

    print(
        f"\nDataset shape: "
        f"{df.shape}"
    )

    # --------------------------------------------------------
    # Validate required graph columns.
    # --------------------------------------------------------

    required_columns = [
        "user_id",
        "merchant_id",
        "device_id",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Required graph columns are missing:\n"
            + "\n".join(
                missing_columns
            )
        )

    # --------------------------------------------------------
    # Load trained artifacts.
    # --------------------------------------------------------

    preprocessor = (
        load_preprocessor()
    )

    model = load_gnn_model(
        device
    )

    # --------------------------------------------------------
    # Generate embeddings.
    # --------------------------------------------------------

    embeddings = (
        generate_embeddings(
            df=df,
            preprocessor=preprocessor,
            model=model,
            device=device,
        )
    )

    # --------------------------------------------------------
    # Save embeddings.
    # --------------------------------------------------------

    save_embeddings(
        df=df,
        embeddings=embeddings,
    )

    # --------------------------------------------------------
    # Final message.
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "GNN EMBEDDING GENERATION COMPLETE"
    )

    print("=" * 70)

    print(
        "\nOutput:"
    )

    print(
        "models/gnn_embeddings.csv"
    )


if __name__ == "__main__":
    main()
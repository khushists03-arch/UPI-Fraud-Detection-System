"""
GNN training pipeline for the UPI Fraud Detection System.

Project-specific design:

    fraud_dataset.csv
            |
            v
    TransactionFeaturePreprocessor
            |
            v
    Transaction node features
            |
            v
    Transaction Graph
            |
            +-- shared user_id
            +-- shared merchant_id
            +-- shared device_id
            |
            v
        2-layer GCN
            |
            +-- 64-dimensional embeddings
            |
            v
      Genuine / Fraud

Important:
- is_fraud is the target and is NEVER used as an input feature.
- user_id, merchant_id and device_id are used for graph construction.
- Preprocessing is fitted only on the training split.
"""

from pathlib import Path
import json
import random

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sklearn.model_selection import train_test_split

from gnn.preprocessing import TransactionFeaturePreprocessor
from gnn.graph_builder import (
    build_transaction_graph,
    normalize_edge_weights,
)
from gnn.gnn_model import GCNClassifier


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Normal location inside the GNN branch.
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "fraud_dataset.csv"
)

# Original project copy.
ORIGINAL_PROJECT_DATA_PATH = (
    PROJECT_ROOT.parent
    / "UPI-Fraud-Detection-System"
    / "data"
    / "raw"
    / "fraud_dataset.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = (
    MODEL_DIR / "gnn_model.pt"
)

PREPROCESSOR_PATH = (
    MODEL_DIR / "gnn_preprocessor.joblib"
)

REFERENCE_PATH = (
    MODEL_DIR / "gnn_reference.csv"
)

EMBEDDINGS_PATH = (
    MODEL_DIR / "gnn_embeddings.csv"
)

METRICS_PATH = (
    MODEL_DIR / "gnn_metrics.json"
)


# ============================================================
# GNN CONFIGURATION
# ============================================================

RANDOM_SEED = 42

HIDDEN_DIM = 64

NUM_CLASSES = 2

DROPOUT = 0.35

LEARNING_RATE = 0.01

WEIGHT_DECAY = 5e-4

MAX_EPOCHS = 40

PATIENCE = 6

VALIDATION_SIZE = 0.15

TEST_SIZE = 0.15


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducible training."""

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# FIND DATASET
# ============================================================

def get_dataset_path() -> Path:
    """
    Find the fraud dataset.

    Priority:

    1. GNN branch:
       data/raw/fraud_dataset.csv

    2. Original project:
       ../UPI-Fraud-Detection-System/data/raw/fraud_dataset.csv
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
# METRICS
# ============================================================

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fraud_probability: np.ndarray,
) -> dict:
    """Calculate fraud classification metrics."""

    result = {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
    }

    if len(np.unique(y_true)) == 2:

        result["roc_auc"] = float(
            roc_auc_score(
                y_true,
                fraud_probability,
            )
        )

    else:

        result["roc_auc"] = None

    return result


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model: GCNClassifier,
    features: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
):
    """Evaluate the GCN on selected transactions."""

    model.eval()

    with torch.no_grad():

        logits, embeddings = model(
            features,
            edge_index,
            edge_weight,
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

    true_labels = (
        labels[mask]
        .cpu()
        .numpy()
    )

    predicted_labels = (
        predictions[mask]
        .cpu()
        .numpy()
    )

    fraud_probability = (
        probabilities[mask, 1]
        .cpu()
        .numpy()
    )

    metrics = calculate_metrics(
        true_labels,
        predicted_labels,
        fraud_probability,
    )

    return (
        metrics,
        predictions,
        probabilities,
        embeddings,
    )


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def main() -> None:

    set_seed()

    print("=" * 70)
    print("UPI FRAUD DETECTION - GNN TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nDevice: {device}")

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    dataset_path = get_dataset_path()

    print(
        f"\nDataset:\n{dataset_path}"
    )

    df = pd.read_csv(
        dataset_path
    )

    print(
        f"\nDataset shape: {df.shape}"
    )

    # --------------------------------------------------------
    # REQUIRED PROJECT COLUMNS
    # --------------------------------------------------------

    required_columns = [
        "is_fraud",
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
            "Required project columns are missing:\n"
            + "\n".join(
                missing_columns
            )
        )

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    df = df.dropna(
        subset=["is_fraud"]
    ).reset_index(
        drop=True
    )

    df["is_fraud"] = (
        pd.to_numeric(
            df["is_fraud"],
            errors="raise",
        )
        .astype(int)
    )

    labels_np = df[
        "is_fraud"
    ].to_numpy(
        dtype=np.int64
    )

    print("\nFraud distribution:")

    print(
        df["is_fraud"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # TRAIN / VALIDATION / TEST
    # --------------------------------------------------------

    all_indices = np.arange(
        len(df)
    )

    train_indices, temporary_indices = (
        train_test_split(
            all_indices,
            test_size=(
                VALIDATION_SIZE
                + TEST_SIZE
            ),
            random_state=RANDOM_SEED,
            stratify=labels_np,
        )
    )

    validation_ratio = (
        VALIDATION_SIZE
        / (
            VALIDATION_SIZE
            + TEST_SIZE
        )
    )

    validation_indices, test_indices = (
        train_test_split(
            temporary_indices,
            test_size=(
                1.0 - validation_ratio
            ),
            random_state=RANDOM_SEED,
            stratify=labels_np[
                temporary_indices
            ],
        )
    )

    num_nodes = len(df)

    train_mask = torch.zeros(
        num_nodes,
        dtype=torch.bool,
    )

    validation_mask = torch.zeros(
        num_nodes,
        dtype=torch.bool,
    )

    test_mask = torch.zeros(
        num_nodes,
        dtype=torch.bool,
    )

    train_mask[
        train_indices
    ] = True

    validation_mask[
        validation_indices
    ] = True

    test_mask[
        test_indices
    ] = True

    print("\nData split:")

    print(
        f"Training:   "
        f"{len(train_indices)}"
    )

    print(
        f"Validation: "
        f"{len(validation_indices)}"
    )

    print(
        f"Testing:    "
        f"{len(test_indices)}"
    )

    # ========================================================
    # PREPROCESSING
    # ========================================================

    print("\n" + "-" * 70)
    print("FEATURE PREPROCESSING")
    print("-" * 70)

    preprocessor = (
        TransactionFeaturePreprocessor()
    )

    # IMPORTANT:
    # Fit ONLY using training transactions.
    preprocessor.fit(
        df.iloc[train_indices]
    )

    # Transform all transactions.
    features_np = (
        preprocessor.transform(df)
    )

    print(
        f"Feature matrix: "
        f"{features_np.shape}"
    )

    print(
        f"GNN input features: "
        f"{preprocessor.input_dim}"
    )

    features = torch.tensor(
        features_np,
        dtype=torch.float32,
    )

    # ========================================================
    # GRAPH
    # ========================================================

    print("\n" + "-" * 70)
    print("TRANSACTION GRAPH")
    print("-" * 70)

    (
        edge_index,
        edge_weight,
        graph_stats,
    ) = build_transaction_graph(
        df
    )

    print(
        f"Nodes: "
        f"{graph_stats['num_nodes']}"
    )

    print(
        f"Unique relationships: "
        f"{graph_stats['num_unique_undirected_relationships']}"
    )

    print(
        f"Edges including self-loops: "
        f"{graph_stats['num_edges_including_self_loops']}"
    )

    print(
        f"Graph entities: "
        f"{graph_stats['entity_columns_used']}"
    )

    normalized_edge_weight = (
        normalize_edge_weights(
            edge_index,
            edge_weight,
            num_nodes,
        )
    )

    # ========================================================
    # MOVE TO DEVICE
    # ========================================================

    features = features.to(
        device
    )

    labels = torch.tensor(
        labels_np,
        dtype=torch.long,
        device=device,
    )

    edge_index = edge_index.to(
        device
    )

    normalized_edge_weight = (
        normalized_edge_weight.to(
            device
        )
    )

    train_mask = train_mask.to(
        device
    )

    validation_mask = (
        validation_mask.to(device)
    )

    test_mask = test_mask.to(
        device
    )

    # ========================================================
    # CREATE GCN
    # ========================================================

    model = GCNClassifier(
        input_dim=preprocessor.input_dim,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    ).to(device)

    print("\n" + "-" * 70)
    print("GCN MODEL")
    print("-" * 70)

    print(model)

    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    training_labels = labels[
        train_mask
    ]

    class_counts = torch.bincount(
        training_labels,
        minlength=2,
    ).float()

    class_weights = (
        class_counts.sum()
        / (
            2.0
            * class_counts.clamp_min(
                1.0
            )
        )
    )

    print("\nTraining classes:")

    print(
        f"Genuine: "
        f"{int(class_counts[0])}"
    )

    print(
        f"Fraud:   "
        f"{int(class_counts[1])}"
    )

    # ========================================================
    # LOSS + OPTIMIZER
    # ========================================================

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ========================================================
    # TRAINING
    # ========================================================

    print("\n" + "=" * 70)
    print("GNN TRAINING")
    print("=" * 70)

    best_validation_f1 = -1.0

    best_state = None

    best_epoch = 0

    epochs_without_improvement = 0

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        model.train()

        optimizer.zero_grad()

        logits, _ = model(
            features,
            edge_index,
            normalized_edge_weight,
        )

        loss = criterion(
            logits[train_mask],
            labels[train_mask],
        )

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        (
            validation_metrics,
            _,
            _,
            _,
        ) = evaluate_model(
            model,
            features,
            edge_index,
            normalized_edge_weight,
            labels,
            validation_mask,
        )

        validation_f1 = (
            validation_metrics["f1"]
        )

        print(
            f"Epoch "
            f"{epoch:02d}/{MAX_EPOCHS} | "
            f"Loss: {loss.item():.4f} | "
            f"Val F1: "
            f"{validation_f1:.4f} | "
            f"Val Recall: "
            f"{validation_metrics['recall']:.4f}"
        )

        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        if validation_f1 > (
            best_validation_f1
        ):

            best_validation_f1 = (
                validation_f1
            )

            best_epoch = epoch

            epochs_without_improvement = 0

            best_state = {
                key: value.detach()
                .cpu()
                .clone()
                for key, value
                in model.state_dict().items()
            }

        else:

            epochs_without_improvement += 1

            if (
                epochs_without_improvement
                >= PATIENCE
            ):

                print(
                    "\nEarly stopping."
                )

                break

    # ========================================================
    # RESTORE BEST MODEL
    # ========================================================

    if best_state is None:

        raise RuntimeError(
            "Training failed to produce "
            "a valid model."
        )

    model.load_state_dict(
        best_state
    )

    model.to(device)

    print(
        f"\nBest epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best validation F1: "
        f"{best_validation_f1:.4f}"
    )

    # ========================================================
    # TEST
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    (
        test_metrics,
        predictions,
        probabilities,
        embeddings,
    ) = evaluate_model(
        model,
        features,
        edge_index,
        normalized_edge_weight,
        labels,
        test_mask,
    )

    print(
        f"\nAccuracy : "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"F1 Score : "
        f"{test_metrics['f1']:.4f}"
    )

    if test_metrics[
        "roc_auc"
    ] is not None:

        print(
            f"ROC-AUC  : "
            f"{test_metrics['roc_auc']:.4f}"
        )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    y_test = labels[
        test_mask
    ].cpu().numpy()

    y_prediction = predictions[
        test_mask
    ].cpu().numpy()

    confusion = confusion_matrix(
        y_test,
        y_prediction,
    )

    print("\nConfusion Matrix:")

    print(confusion)

    # ========================================================
    # SAVE ARTIFACTS
    # ========================================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Model checkpoint
    # --------------------------------------------------------

    checkpoint = {
        "model_state_dict":
            model.state_dict(),

        "input_dim":
            preprocessor.input_dim,

        "hidden_dim":
            HIDDEN_DIM,

        "num_classes":
            NUM_CLASSES,

        "dropout":
            DROPOUT,

        "feature_names":
            preprocessor.feature_names_,

        "best_epoch":
            best_epoch,

        "validation_f1":
            best_validation_f1,

        "random_seed":
            RANDOM_SEED,
    }

    torch.save(
        checkpoint,
        MODEL_PATH,
    )

    print(
        f"\nSaved GNN model:"
        f"\n{MODEL_PATH}"
    )

    # --------------------------------------------------------
    # Preprocessor
    # --------------------------------------------------------

    joblib.dump(
        preprocessor,
        PREPROCESSOR_PATH,
    )

    print(
        f"\nSaved preprocessor:"
        f"\n{PREPROCESSOR_PATH}"
    )

    # --------------------------------------------------------
    # Reference transactions
    # --------------------------------------------------------

    reference_df = df.copy()

    reference_df.insert(
        0,
        "gnn_node_index",
        np.arange(
            len(reference_df)
        ),
    )

    reference_df.to_csv(
        REFERENCE_PATH,
        index=False,
    )

    print(
        f"\nSaved reference data:"
        f"\n{REFERENCE_PATH}"
    )

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    embeddings_np = (
        embeddings
        .detach()
        .cpu()
        .numpy()
    )

    embedding_columns = [
        f"embedding_{index}"
        for index in range(
            embeddings_np.shape[1]
        )
    ]

    embeddings_df = pd.DataFrame(
        embeddings_np,
        columns=embedding_columns,
    )

    embeddings_df.insert(
        0,
        "gnn_node_index",
        np.arange(
            len(embeddings_df)
        ),
    )

    embeddings_df[
        "is_fraud"
    ] = labels_np

    embeddings_df.to_csv(
        EMBEDDINGS_PATH,
        index=False,
    )

    print(
        f"\nSaved embeddings:"
        f"\n{EMBEDDINGS_PATH}"
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = {
        "dataset": str(
            dataset_path
        ),

        "num_transactions":
            num_nodes,

        "num_features":
            preprocessor.input_dim,

        "hidden_dim":
            HIDDEN_DIM,

        "dropout":
            DROPOUT,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "best_epoch":
            best_epoch,

        "validation_f1":
            best_validation_f1,

        "graph":
            graph_stats,

        "test_metrics":
            test_metrics,

        "confusion_matrix":
            confusion.tolist(),
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    print(
        f"\nSaved metrics:"
        f"\n{METRICS_PATH}"
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 70)
    print("GNN TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"\nTest F1: "
        f"{test_metrics['f1']:.4f}"
    )

    if test_metrics[
        "roc_auc"
    ] is not None:

        print(
            f"Test ROC-AUC: "
            f"{test_metrics['roc_auc']:.4f}"
        )

    print("\nArtifacts created:")

    print(
        "  models/gnn_model.pt"
    )

    print(
        "  models/gnn_preprocessor.joblib"
    )

    print(
        "  models/gnn_reference.csv"
    )

    print(
        "  models/gnn_embeddings.csv"
    )

    print(
        "  models/gnn_metrics.json"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
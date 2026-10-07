"""
Train the GNN for UPI Fraud Detection.
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


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "fraud_dataset.csv"
)

# Fallback dataset location
ORIGINAL_DATA_PATH = (
    PROJECT_ROOT.parent
    / "UPI-Fraud-Detection-System"
    / "data"
    / "raw"
    / "fraud_dataset.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "gnn_model.pt"
PREPROCESSOR_PATH = MODEL_DIR / "gnn_preprocessor.joblib"
REFERENCE_PATH = MODEL_DIR / "gnn_reference.csv"
EMBEDDINGS_PATH = MODEL_DIR / "gnn_embeddings.csv"
METRICS_PATH = MODEL_DIR / "gnn_metrics.json"


# GNN settings
SEED = 42
HIDDEN_DIM = 64
DROPOUT = 0.35
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
MAX_EPOCHS = 40
PATIENCE = 6


def set_seed():
    """Make training results reproducible."""

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def get_dataset_path():
    """Find the fraud dataset."""

    if DATA_PATH.exists():
        return DATA_PATH

    if ORIGINAL_DATA_PATH.exists():
        return ORIGINAL_DATA_PATH

    raise FileNotFoundError(
        "fraud_dataset.csv was not found."
    )


def calculate_metrics(
    y_true,
    y_pred,
    fraud_probability,
):
    """Calculate model performance."""

    metrics = {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
    }

    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(
            y_true,
            fraud_probability,
        )
    else:
        metrics["roc_auc"] = None

    return metrics


def evaluate(
    model,
    features,
    edge_index,
    edge_weight,
    labels,
    mask,
):
    """Evaluate the GNN on a selected data split."""

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

    y_true = labels[mask].cpu().numpy()

    y_pred = predictions[mask].cpu().numpy()

    fraud_probability = (
        probabilities[mask, 1]
        .cpu()
        .numpy()
    )

    metrics = calculate_metrics(
        y_true,
        y_pred,
        fraud_probability,
    )

    return (
        metrics,
        predictions,
        probabilities,
        embeddings,
    )


def main():

    set_seed()

    print("=" * 60)
    print("UPI FRAUD DETECTION - GNN TRAINING")
    print("=" * 60)

    # Select CPU or GPU.
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nDevice: {device}")

    # Load dataset.
    dataset_path = get_dataset_path()

    print(f"\nDataset:\n{dataset_path}")

    df = pd.read_csv(dataset_path)

    print(f"Dataset shape: {df.shape}")

    # Required project columns.
    required_columns = [
        "is_fraud",
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

    # Prepare target.
    df = df.dropna(
        subset=["is_fraud"]
    ).reset_index(drop=True)

    df["is_fraud"] = pd.to_numeric(
        df["is_fraud"],
        errors="raise",
    ).astype(int)

    labels_np = df[
        "is_fraud"
    ].to_numpy(dtype=np.int64)

    # --------------------------------------------------
    # Split data into training, validation and testing.
    # --------------------------------------------------

    indices = np.arange(len(df))

    train_indices, temp_indices = train_test_split(
        indices,
        test_size=0.30,
        random_state=SEED,
        stratify=labels_np,
    )

    validation_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.50,
        random_state=SEED,
        stratify=labels_np[temp_indices],
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

    train_mask[train_indices] = True
    validation_mask[validation_indices] = True
    test_mask[test_indices] = True

    print("\nData split:")
    print(f"Training:   {len(train_indices)}")
    print(f"Validation: {len(validation_indices)}")
    print(f"Testing:    {len(test_indices)}")

    # --------------------------------------------------
    # Preprocessing
    # --------------------------------------------------

    print("\nPreparing features...")

    preprocessor = (
        TransactionFeaturePreprocessor()
    )

    # Fit only on training data.
    preprocessor.fit(
        df.iloc[train_indices]
    )

    # Transform all transactions.
    features_np = preprocessor.transform(df)

    print(
        f"Feature matrix: {features_np.shape}"
    )

    features = torch.tensor(
        features_np,
        dtype=torch.float32,
    )

    # --------------------------------------------------
    # Build transaction graph.
    # --------------------------------------------------

    print("\nBuilding transaction graph...")

    (
        edge_index,
        edge_weight,
        graph_stats,
    ) = build_transaction_graph(df)

    print(
        f"Nodes: "
        f"{graph_stats['num_nodes']}"
    )

    print(
        f"Relationships: "
        f"{graph_stats['num_unique_undirected_relationships']}"
    )

    print(
        f"Edges: "
        f"{graph_stats['num_edges_including_self_loops']}"
    )

    # Normalize graph weights.
    edge_weight = normalize_edge_weights(
        edge_index,
        edge_weight,
        num_nodes,
    )

    # Move data to device.
    features = features.to(device)

    labels = torch.tensor(
        labels_np,
        dtype=torch.long,
        device=device,
    )

    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    train_mask = train_mask.to(device)
    validation_mask = validation_mask.to(device)
    test_mask = test_mask.to(device)

    # --------------------------------------------------
    # Create GNN.
    # --------------------------------------------------

    model = GCNClassifier(
        input_dim=preprocessor.input_dim,
        hidden_dim=HIDDEN_DIM,
        num_classes=2,
        dropout=DROPOUT,
    ).to(device)

    print("\nGNN model:")
    print(model)

    # --------------------------------------------------
    # Give more weight to the fraud class.
    # --------------------------------------------------

    training_labels = labels[train_mask]

    class_counts = torch.bincount(
        training_labels,
        minlength=2,
    ).float()

    class_weights = (
        class_counts.sum()
        / (
            2
            * class_counts.clamp_min(1)
        )
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------
    # Train the GNN.
    # --------------------------------------------------

    best_f1 = -1
    best_state = None
    best_epoch = 0
    patience_count = 0

    print("\nTraining GNN...")

    for epoch in range(1, MAX_EPOCHS + 1):

        model.train()

        optimizer.zero_grad()

        logits, _ = model(
            features,
            edge_index,
            edge_weight,
        )

        loss = criterion(
            logits[train_mask],
            labels[train_mask],
        )

        loss.backward()
        optimizer.step()

        validation_metrics, _, _, _ = evaluate(
            model,
            features,
            edge_index,
            edge_weight,
            labels,
            validation_mask,
        )

        val_f1 = validation_metrics["f1"]

        print(
            f"Epoch {epoch:02d} | "
            f"Loss: {loss.item():.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        # Save the best model.
        if val_f1 > best_f1:

            best_f1 = val_f1
            best_epoch = epoch
            patience_count = 0

            best_state = {
                key: value.detach()
                .cpu()
                .clone()
                for key, value
                in model.state_dict().items()
            }

        else:

            patience_count += 1

            if patience_count >= PATIENCE:

                print("\nEarly stopping.")
                break

    # Restore best model.
    model.load_state_dict(best_state)
    model.to(device)

    print(
        f"\nBest epoch: {best_epoch}"
    )

    # --------------------------------------------------
    # Test the final model.
    # --------------------------------------------------

    (
        test_metrics,
        predictions,
        probabilities,
        embeddings,
    ) = evaluate(
        model,
        features,
        edge_index,
        edge_weight,
        labels,
        test_mask,
    )

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    print(
        f"Accuracy : {test_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: {test_metrics['precision']:.4f}"
    )

    print(
        f"Recall   : {test_metrics['recall']:.4f}"
    )

    print(
        f"F1 Score : {test_metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC  : {test_metrics['roc_auc']:.4f}"
    )

    # Confusion matrix.
    y_test = labels[
        test_mask
    ].cpu().numpy()

    y_pred = predictions[
        test_mask
    ].cpu().numpy()

    confusion = confusion_matrix(
        y_test,
        y_pred,
    )

    print("\nConfusion Matrix:")
    print(confusion)

    # --------------------------------------------------
    # Save model and supporting files.
    # --------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_state_dict":
            model.state_dict(),

        "input_dim":
            preprocessor.input_dim,

        "hidden_dim":
            HIDDEN_DIM,

        "num_classes":
            2,

        "dropout":
            DROPOUT,

        "best_epoch":
            best_epoch,

        "validation_f1":
            best_f1,
    }

    torch.save(
        checkpoint,
        MODEL_PATH,
    )

    joblib.dump(
        preprocessor,
        PREPROCESSOR_PATH,
    )

    # Save transaction references.
    reference_df = df.copy()

    reference_df.insert(
        0,
        "gnn_node_index",
        np.arange(len(df)),
    )

    reference_df.to_csv(
        REFERENCE_PATH,
        index=False,
    )

    # Save embeddings.
    embeddings_np = (
        embeddings.cpu().numpy()
    )

    embedding_columns = [
        f"embedding_{i}"
        for i in range(
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
        np.arange(len(df)),
    )

    embeddings_df["is_fraud"] = labels_np

    embeddings_df.to_csv(
        EMBEDDINGS_PATH,
        index=False,
    )

    # Save metrics.
    metrics = {
        "dataset": str(dataset_path),
        "num_transactions": num_nodes,
        "num_features":
            preprocessor.input_dim,
        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
        "learning_rate": LEARNING_RATE,
        "best_epoch": best_epoch,
        "validation_f1": best_f1,
        "graph": graph_stats,
        "test_metrics": test_metrics,
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

    print("\n" + "=" * 60)
    print("GNN TRAINING COMPLETE")
    print("=" * 60)

    print("\nCreated files:")
    print("models/gnn_model.pt")
    print("models/gnn_preprocessor.joblib")
    print("models/gnn_reference.csv")
    print("models/gnn_embeddings.csv")
    print("models/gnn_metrics.json")


if __name__ == "__main__":
    main()
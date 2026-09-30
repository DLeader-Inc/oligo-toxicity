"""Same-split comparison utilities for OligoGym and project baselines."""

from __future__ import annotations

import ast
import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split
from sklearn.preprocessing import normalize
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

from oligogym.features import KMersCounts, OneHotEncoder

from src.benchmarking.regression_evaluation import regression_metrics as _regression_metrics
from src.data.grouping import canonical_sequence_groups
from src.features.featurizer import featurize_dataset


@dataclass(frozen=True)
class EndpointSource:
    project_key: str
    paper_dataset: str
    benchmark_dataset: str
    resource_file: str


ENDPOINT_SOURCES: tuple[EndpointSource, ...] = (
    EndpointSource("cytotoxicity", "Cytotox LNA", "cytotox_lna", "Papargyri_2020_1_processed.csv.gz"),
    EndpointSource("neurotoxicity_lna", "Neurotox LNA", "acute_neurotox_lna", "Hagedorn_2022_1_processed.csv.gz"),
    EndpointSource("neurotoxicity_moe", "Neurotox MOE", "acute_neurotox_moe_cleaned", "MOE_Neurotox_1_processed.csv.gz"),
    EndpointSource("tlr7_immunotox", "TLR7", "immune_modulation_TLR7", "Alharbi_2020_1_processed.csv.gz"),
    EndpointSource("tlr8_immunotox", "TLR8", "immune_modulation_TLR8", "Alharbi_2020_2_processed.csv.gz"),
)


@dataclass(frozen=True)
class SplitDefinition:
    split_mode: str
    split_id: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    group_labels: np.ndarray


def regression_metrics(
    y: np.ndarray,
    prediction: np.ndarray,
    *,
    normalization_std: float | None = None,
) -> dict[str, float]:
    """Return mandatory ranking, fit, and normalized-error metrics."""

    return _regression_metrics(y, prediction, normalization_std=normalization_std)


def make_nucleobase_splits(
    helms: np.ndarray,
    y: np.ndarray,
    *,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    test_size: float = 0.2,
    n_clusters: int = 8,
) -> Iterator[SplitDefinition]:
    """Recreate OligoGym's k-mer/KMeans/GroupShuffleSplit protocol."""

    kmer = KMersCounts(k=[1, 2, 3])
    vectors = np.asarray(kmer.fit_transform(helms))
    vectors = normalize(vectors, norm="l2")
    for split_id, seed in enumerate(seeds, start=1):
        labels = KMeans(
            n_clusters=n_clusters,
            random_state=seed,
            n_init="auto",
        ).fit_predict(vectors)
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=seed,
        )
        train_indices, test_indices = next(splitter.split(helms, y, labels))
        yield SplitDefinition(
            split_mode="nucleobase_holdout",
            split_id=split_id,
            train_indices=np.asarray(train_indices),
            test_indices=np.asarray(test_indices),
            group_labels=np.asarray(labels),
        )


def make_exact_sequence_splits(
    frame: pd.DataFrame,
    y: np.ndarray,
    *,
    dataset_key: str,
    n_splits: int = 5,
) -> Iterator[SplitDefinition]:
    """Create the project's canonical exact-molecular-sequence GroupKFold."""

    groups = canonical_sequence_groups(frame, dataset_key=dataset_key).to_numpy()
    splitter = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    dummy = np.zeros((len(frame), 1), dtype=np.float32)
    for split_id, (train_indices, test_indices) in enumerate(
        splitter.split(dummy, y, groups), start=1
    ):
        yield SplitDefinition(
            split_mode="exact_sequence_groupkfold",
            split_id=split_id,
            train_indices=np.asarray(train_indices),
            test_indices=np.asarray(test_indices),
            group_labels=groups,
        )


def parse_config(value: object) -> dict[str, object]:
    """Parse a dictionary serialized in the public benchmark CSV."""

    if isinstance(value, dict):
        return dict(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, dict):
        raise TypeError(f"expected serialized dict, got {type(parsed).__name__}")
    return parsed


def fit_oligogym_features(
    train_helms: np.ndarray,
    test_helms: np.ndarray,
    *,
    featurizer_name: str,
    featurizer_args: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit an official OligoGym featurizer on outer-train only."""

    if featurizer_name == "OneHotEncoder":
        featurizer = OneHotEncoder(**featurizer_args)
    elif featurizer_name == "KMersCounts":
        featurizer = KMersCounts(**featurizer_args)
    else:
        raise ValueError(f"unsupported published featurizer: {featurizer_name}")
    x_train = np.asarray(featurizer.fit_transform(train_helms), dtype=np.float32)
    x_test = np.asarray(featurizer.transform(test_helms), dtype=np.float32)
    return x_train, x_test


class OligoGymCNN(nn.Module):
    """Architecture-equivalent implementation of OligoGym's CNN regressor."""

    def __init__(
        self,
        input_dim: int,
        sequence_length: int,
        *,
        hidden_dim: int,
        kernel_size: int,
        depth: int,
        pooling_operation: str,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = input_dim
        length = sequence_length
        for _ in range(depth):
            layers.append(nn.Conv1d(channels, hidden_dim, kernel_size=kernel_size))
            layers.append(nn.ReLU())
            layers.append(nn.AvgPool1d(2) if pooling_operation == "avg" else nn.MaxPool1d(2))
            channels = hidden_dim
            length = (length - kernel_size + 1) // 2
        if length < 1:
            raise ValueError("published CNN recipe collapses the sequence dimension")
        self.encoder = nn.Sequential(*layers)
        self.output = nn.Linear(channels * length, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x.transpose(1, 2))
        return self.output(x.flatten(start_dim=1)).squeeze(-1)


class OligoGymTransformer(nn.Module):
    """Architecture-equivalent implementation of OligoGym's Transformer."""

    def __init__(
        self,
        input_dim: int,
        sequence_length: int,
        *,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(d_model, 1)
        position = torch.arange(sequence_length, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-np.log(10000.0) / d_model)
        )
        encoding = torch.zeros(sequence_length, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("positional_encoding", encoding.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = x + self.positional_encoding[:, : x.shape[1], :]
        x = self.encoder(x).mean(dim=1)
        return self.output(self.dropout(x)).squeeze(-1)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_predict_torch_recipe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    model_name: str,
    model_args: dict[str, object],
    seed: int,
    max_epochs: int = 100,
    patience: int = 5,
    batch_size: int = 32,
    device: str | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Train the published neural recipe with the paper's optimizer settings."""

    if x_train.ndim != 3:
        raise ValueError(f"{model_name} requires position-wise 3D features")
    _seed_everything(seed)
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sequence_length = int(x_train.shape[1])
    input_dim = int(x_train.shape[2])
    if model_name == "CNN":
        model = OligoGymCNN(
            input_dim,
            sequence_length,
            hidden_dim=int(model_args["hidden_dim"]),
            kernel_size=int(model_args["kernel_size"]),
            depth=int(model_args["depth"]),
            pooling_operation=str(model_args["pooling_operation"]),
        )
    elif model_name == "Transformer":
        model = OligoGymTransformer(
            input_dim,
            sequence_length,
            d_model=int(model_args["d_model"]),
            nhead=int(model_args["nhead"]),
            num_layers=int(model_args["num_layers"]),
            dim_feedforward=int(model_args["dim_feedforward"]),
            dropout=float(model_args["dropout"]),
        )
    else:
        raise ValueError(f"unsupported published neural model: {model_name}")
    model = model.to(selected_device)

    row_indices = np.arange(len(x_train))
    fit_indices, val_indices = train_test_split(
        row_indices,
        test_size=0.2,
        random_state=42,
    )
    generator = torch.Generator().manual_seed(seed)
    train_dataset = TensorDataset(
        torch.from_numpy(x_train[fit_indices]).float(),
        torch.from_numpy(y_train[fit_indices]).float(),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(x_train[val_indices]).float(),
        torch.from_numpy(y_train[val_indices]).float(),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    loss_function = nn.MSELoss()
    best_val = float("inf")
    epochs_without_improvement = 0
    epochs_trained = 0
    last_val = float("nan")
    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(selected_device)
            yb = yb.to(selected_device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        total_loss = 0.0
        total_rows = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(selected_device)
                yb = yb.to(selected_device)
                batch_loss = loss_function(model(xb), yb)
                total_loss += float(batch_loss.item()) * len(xb)
                total_rows += len(xb)
        last_val = total_loss / max(total_rows, 1)
        epochs_trained = epoch + 1
        if last_val < best_val:
            best_val = last_val
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x_test), 512):
            xb = torch.from_numpy(x_test[start : start + 512]).float().to(selected_device)
            predictions.append(model(xb).cpu().numpy())
    metadata = {
        "device": selected_device,
        "epochs_trained": epochs_trained,
        "best_validation_mse": best_val,
        "last_validation_mse": last_val,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    return np.concatenate(predictions).astype(float), metadata


def fit_predict_published_recipe(
    train_helms: np.ndarray,
    y_train: np.ndarray,
    test_helms: np.ndarray,
    *,
    model_name: str,
    model_args: dict[str, object],
    featurizer_name: str,
    featurizer_args: dict[str, object],
    seed: int,
    n_jobs: int = 8,
    device: str | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit one fixed published OligoGym recipe on an outer-train split."""

    x_train, x_test = fit_oligogym_features(
        train_helms,
        test_helms,
        featurizer_name=featurizer_name,
        featurizer_args=featurizer_args,
    )
    metadata: dict[str, object] = {
        "feature_shape": "x".join(str(value) for value in x_train.shape[1:]),
    }
    if model_name in {"XGBoostModel", "XGB"}:
        if x_train.ndim > 2:
            x_train = x_train.reshape(len(x_train), -1)
            x_test = x_test.reshape(len(x_test), -1)
        model = XGBRegressor(
            **model_args,
            random_state=seed,
            n_jobs=n_jobs,
            verbosity=0,
        )
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
    elif model_name in {"RandomForestModel", "RF"}:
        if x_train.ndim > 2:
            x_train = x_train.reshape(len(x_train), -1)
            x_test = x_test.reshape(len(x_test), -1)
        model = RandomForestRegressor(
            **model_args,
            random_state=seed,
            n_jobs=n_jobs,
        )
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
    elif model_name in {"CNN", "Transformer"}:
        prediction, neural_metadata = fit_predict_torch_recipe(
            x_train,
            y_train,
            x_test,
            model_name=model_name,
            model_args=model_args,
            seed=seed,
            device=device,
        )
        metadata.update(neural_metadata)
    else:
        raise ValueError(f"unsupported comparison model: {model_name}")
    return np.asarray(prediction, dtype=float).reshape(-1), metadata


def current_baseline_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Build the exact feature matrix used by the current core baseline."""

    features, parsed = featurize_dataset(frame)
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return features.to_numpy(dtype=np.float32), parsed


def fit_predict_current_baseline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    seed: int,
    n_jobs: int = 8,
) -> np.ndarray:
    """Fit the project's fixed core baseline XGBoost recipe."""

    model = XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        verbosity=0,
        n_jobs=n_jobs,
        tree_method="hist",
    )
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_test), dtype=float)


def fit_predict_project_xgb_features(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    seed: int,
    n_jobs: int = 8,
    device: str | None = None,
) -> np.ndarray:
    """Fit the fixed project XGBoost on an already-fitted feature view."""

    if x_train.ndim > 2:
        x_train = x_train.reshape(len(x_train), -1)
        x_test = x_test.reshape(len(x_test), -1)
    kwargs: dict[str, object] = {
        "n_estimators": 600,
        "max_depth": 4,
        "learning_rate": 0.02,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 3.0,
        "reg_alpha": 0.1,
        "random_state": seed,
        "verbosity": 0,
        "n_jobs": n_jobs,
        "tree_method": "hist",
    }
    if device is not None:
        kwargs["device"] = device
    model = XGBRegressor(**kwargs)
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_test), dtype=float)


def clone_recipe_row(row: pd.Series) -> dict[str, object]:
    """Convert one reconstructed public configuration to a plain mapping."""

    return {
        "model": str(row["model"]),
        "model_args": copy.deepcopy(parse_config(row["model_args"])),
        "featurizer": str(row["featurizer"]),
        "featurizer_args": copy.deepcopy(parse_config(row["featurizer_args"])),
        "published_pcc_mean": float(row["pcc_mean"]),
        "published_pcc_std": float(row["pcc_std"]),
    }


def load_endpoint_frame(official_root: Path, source: EndpointSource) -> tuple[pd.DataFrame, Path]:
    """Load the exact packaged CSV used by the public benchmark."""

    path = official_root / "oligogym/resources/pkg_dataset" / source.resource_file
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {"x", "y", "targets", "fasta", "smiles"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    return frame, path

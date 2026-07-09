from dataclasses import dataclass
from pathlib import Path
from scipy.sparse import csr_matrix
import pandas as pd


@dataclass(frozen=True)
class DataTransformationArtifact:

    train_data_path: Path

    test_data_path: Path

    preprocessor_path: Path

    X_train: csr_matrix

    X_test: csr_matrix

    y_train: pd.Series

    y_test: pd.Series


@dataclass(frozen=True)
class ModelTrainerArtifact:

    trained_model_path: Path

    metrics: dict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:

    root_dir: Path

    source_url: str

    source_data_file: Path

    local_data_file: Path


@dataclass(frozen=True)
class DataValidationConfig:

    root_dir: Path

    STATUS_FILE: Path

    data_file_path: Path

    all_schema: dict

@dataclass(frozen=True)
class DataTransformationConfig:

    root_dir: Path

    data_file_path: Path

    preprocessor_path: Path

    train_data_path: Path

    test_data_path: Path

    text_columns: list

    target_column: str

    tfidf_params: dict

@dataclass(frozen=True)
class DataTransformationArtifact:

    train_data_path: Path

    test_data_path: Path

    preprocessor_path: Path

    y_train: pd.Series

    y_test: pd.Series
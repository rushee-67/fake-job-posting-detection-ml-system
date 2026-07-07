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
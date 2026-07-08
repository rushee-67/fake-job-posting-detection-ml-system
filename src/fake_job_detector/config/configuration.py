from fake_job_detector.constants import (
    CONFIG_FILE_PATH,
    PARAMS_FILE_PATH,
    SCHEMA_FILE_PATH
)

from pathlib import Path
from fake_job_detector.utils.common import read_yaml, create_directories
from fake_job_detector.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig


class ConfigurationManager:

    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
        schema_filepath=SCHEMA_FILE_PATH,
    ):

        self.config = read_yaml(config_filepath)

        create_directories([Path(self.config.artifacts_root)])

        self.params = read_yaml(params_filepath)

        self.schema = read_yaml(schema_filepath)
        

    def get_data_ingestion_config(self) -> DataIngestionConfig:

        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_url=config.source_url,
            source_data_file=Path(config.source_data_file),
            local_data_file=Path(config.local_data_file)
        )

        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:

        config = self.config.data_validation

        create_directories([Path(config.root_dir)])

        data_validation_config = DataValidationConfig(

            root_dir=Path(config.root_dir),

            STATUS_FILE=Path(config.STATUS_FILE),

            data_file_path=Path(config.data_file_path),

            all_schema=self.schema
        )

        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:

        config = self.config.data_transformation

        create_directories([Path(config.root_dir)])

        data_transformation_config = DataTransformationConfig(

            root_dir=Path(config.root_dir),

            data_file_path=Path(config.data_file_path),

            preprocessor_path=Path(config.preprocessor_path),

            train_data_path=Path(config.train_data_path),

            test_data_path=Path(config.test_data_path),

            text_columns=self.schema.TEXT_COLUMNS,

            target_column=self.schema.TARGET_COLUMN,

            tfidf_params=self.params.tfidf
        )

        return data_transformation_config

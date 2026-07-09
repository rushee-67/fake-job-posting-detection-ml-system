import sys
import pandas as pd

from fake_job_detector.logger import logger
from fake_job_detector.exception import CustomException
from fake_job_detector.entity.config_entity import DataValidationConfig


class DataValidation:

    def __init__(self, config: DataValidationConfig):
        self.config = config


    def initiate_data_validation(self) -> bool:

        try:
            logger.info("Starting data validation...")

            validation_status = True

            # Read dataset
            df = pd.read_csv(self.config.data_file_path)

            # Required columns from schema
            required_columns = set(self.config.all_schema.TEXT_COLUMNS)
            required_columns.add(self.config.all_schema.TARGET_COLUMN)

            # Check missing columns
            missing_columns = list(required_columns - set(df.columns))

            if missing_columns:
                validation_status = False
                logger.error(f"Missing columns: {missing_columns}")

            # Check empty dataset
            if df.empty:
                validation_status = False
                logger.error("Dataset is empty.")

            # Check duplicate column names
            if df.columns.duplicated().any():
                validation_status = False
                logger.error("Duplicate column names found.")

            # Write validation status
            with open(self.config.status_file, "w") as f:
                if validation_status:
                    f.write("Validation Status: True\n")
                else:
                    f.write("Validation Status: False\n")

                    if missing_columns:
                        f.write(
                            f"Missing Columns: {missing_columns}\n"
                        )

            logger.info(
                f"Data Validation completed with status: {validation_status}"
            )

            return validation_status

        except Exception as e:
            raise CustomException(e, sys)
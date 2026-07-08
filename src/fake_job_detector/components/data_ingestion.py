from pathlib import Path
import shutil
import sys

from fake_job_detector.logger import logger
from fake_job_detector.exception import CustomException
from fake_job_detector.entity.config_entity import DataIngestionConfig

class DataIngestion:

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self):

        try:

            source_file = Path("data/fake_job_postings.csv")

            destination_file = self.config.local_data_file

            shutil.copy(
                self.config.source_data_file,
                self.config.local_data_file
            )

            logger.info("Data ingestion completed successfully.")

            return destination_file

        except Exception as e:
            raise CustomException(e, sys)
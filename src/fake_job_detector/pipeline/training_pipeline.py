from fake_job_detector.config.configuration import ConfigurationManager
from fake_job_detector.components.data_ingestion import DataIngestion
from fake_job_detector.components.data_validation import DataValidation


class TrainingPipeline:

    def __init__(self):
        pass

    def run(self):

        config = ConfigurationManager()

        data_ingestion_config = config.get_data_ingestion_config()

        data_ingestion = DataIngestion(data_ingestion_config)

        data_ingestion.initiate_data_ingestion()

        data_validation_config = config.get_data_validation_config()

        data_validation = DataValidation(data_validation_config)

        data_validation.initiate_data_validation()
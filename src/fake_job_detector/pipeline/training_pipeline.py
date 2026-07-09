from fake_job_detector.config.configuration import ConfigurationManager
from fake_job_detector.components.data_ingestion import DataIngestion
from fake_job_detector.components.data_validation import DataValidation
from fake_job_detector.components.data_transformation import DataTransformation
from fake_job_detector.components.model_trainer import ModelTrainer


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

        data_transformation_config = config.get_data_transformation_config()

        data_transformation = DataTransformation(data_transformation_config)

        data_transformation_artifact = data_transformation.initiate_data_transformation()

        model_trainer_config = config.get_model_trainer_config()

        model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact)

        model_trainer_artifact = model_trainer.initiate_model_training()
from fake_job_detector.config.configuration import ConfigurationManager
from fake_job_detector.components.prediction import Prediction

class PredictionPipeline:

    def __init__(self):

        config = ConfigurationManager()

        prediction_config = (
            config.get_prediction_pipeline_config()
        )

        self.prediction = Prediction(
            prediction_config
        )

    def predict(
        self,
        title,
        company_profile,
        description,
        requirements,
        benefits
    ):

        return self.prediction.predict(
            title,
            company_profile,
            description,
            requirements,
            benefits
        )
from fake_job_detector.entity.config_entity import (
    PredictionPipelineConfig
)

from fake_job_detector.utils.common import load_object
from fake_job_detector.utils.text_preprocessor import TextPreprocessor


class Prediction:

    def __init__(
        self,
        config: PredictionPipelineConfig
    ):

        self.config = config

        self.preprocessor = TextPreprocessor(
            text_columns=[]
        )

    def load_artifacts(self):

        vectorizer = load_object(
            self.config.preprocessor_path
        )

        model = load_object(
            self.config.model_path
        )

        return vectorizer, model
    
    def predict(
    self,
    title,
    company_profile,
    description,
    requirements,
    benefits
    ):

        vectorizer, model = self.load_artifacts()

        text = self.preprocessor.create_text(
            title,
            company_profile,
            description,
            requirements,
            benefits
        )

        text = self.preprocessor.clean_text(text)

        text_vector = vectorizer.transform([text])

        prediction = model.predict(text_vector)

        return (
            "Fraudulent Job"
            if prediction[0] == 1
            else "Genuine Job"
        )
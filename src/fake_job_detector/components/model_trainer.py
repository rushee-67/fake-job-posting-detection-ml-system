import joblib
import sys
import time
import mlflow

from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

import mlflow
import mlflow.sklearn

from fake_job_detector.logger import logger
from fake_job_detector.exception import CustomException
from fake_job_detector.entity.config_entity import ModelTrainerConfig
from fake_job_detector.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from fake_job_detector.utils.common import save_object
from fake_job_detector import mlflow_config


class ModelTrainer:

    def __init__(
        self,
        config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact
    ):

        self.config = config

        self.data_artifact = data_transformation_artifact


    def train_model(self):

        logger.info("Training Linear SVM...")

        start = time.time()

        model = LinearSVC(
            C=self.config.svm_params.C,
            class_weight=self.config.svm_params.class_weight,
            random_state=self.config.svm_params.random_state
        )

        model.fit(
            self.data_artifact.X_train,
            self.data_artifact.y_train
        )

        training_time = time.time() - start

        return model, training_time
    
    
    def evaluate_model(self, model, training_time):

        logger.info("Evaluating model...")

        y_pred = model.predict(self.data_artifact.X_test)

        # Decision scores for ROC-AUC
        decision_scores = model.decision_function(
            self.data_artifact.X_test
        )

        metrics = {

            "accuracy": accuracy_score(
                self.data_artifact.y_test,
                y_pred
            ),

            "precision": precision_score(
                self.data_artifact.y_test,
                y_pred
            ),

            "recall": recall_score(
                self.data_artifact.y_test,
                y_pred
            ),

            "f1_score": f1_score(
                self.data_artifact.y_test,
                y_pred
            ),

            "roc_auc": roc_auc_score(
                self.data_artifact.y_test,
                decision_scores
            ),

            "training_time": training_time
        }

        logger.info(f"Evaluation Metrics: {metrics}")

        return metrics
    
        
    def save_model(self, model):

        save_object(
            self.config.trained_model_path,
            model
        )

        logger.info(
            f"Model saved at {self.config.trained_model_path}"
        )

    def initiate_model_training(self):

        try:

            logger.info("Starting model training...")

            with mlflow.start_run():

                # Train model
                model, training_time = self.train_model()

                # Evaluate model
                metrics = self.evaluate_model(
                    model,
                    training_time
                )

                # Log parameters
                mlflow.log_params(dict(self.config.svm_params))

                # Log metrics
                mlflow.log_metrics(metrics)

                # Log model
                mlflow.sklearn.log_model(
                    sk_model=model,
                    name="linear_svm"
                )

                # Save model locally
                self.save_model(model)

                logger.info("Model training completed successfully.")

                return ModelTrainerArtifact(

                    trained_model_path=self.config.trained_model_path,

                    metrics=metrics
                )

        except Exception as e:
            raise CustomException(e, sys)
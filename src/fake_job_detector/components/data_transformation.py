import re
import string
import sys
import unicodedata
import joblib
import pandas as pd

from bs4 import BeautifulSoup
from scipy.sparse import save_npz

from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from fake_job_detector.logger import logger
from fake_job_detector.exception import CustomException
from fake_job_detector.entity.config_entity import DataTransformationConfig
from fake_job_detector.entity.artifact_entity import DataTransformationArtifact
from fake_job_detector.utils.common import save_object
from fake_job_detector.utils.text_preprocessor import TextPreprocessor


class DataTransformation:

    def __init__(self, config: DataTransformationConfig):

        self.config = config

        self.preprocessor = TextPreprocessor(
            self.config.text_columns
        )

    

    def _split_data(self, df):

        X = df["text"]

        y = df[self.config.target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        logger.info("Train-test split completed.")

        return X_train, X_test, y_train, y_test
    
    def _fit_vectorizer(self, X_train, X_test):

        vectorizer = TfidfVectorizer(
            max_features=self.config.tfidf_params.max_features,
            ngram_range=tuple(self.config.tfidf_params.ngram_range),
            min_df=self.config.tfidf_params.min_df,
            max_df=self.config.tfidf_params.max_df
        )

        X_train_tfidf = vectorizer.fit_transform(X_train)

        X_test_tfidf = vectorizer.transform(X_test)

        logger.info("TF-IDF vectorization completed.")

        return (
            X_train_tfidf,
            X_test_tfidf,
            vectorizer
        )

    def _save_preprocessor(self, vectorizer):

        try:

            save_object(
                self.config.preprocessor_path,
                vectorizer
            )

            logger.info(
                f"Preprocessor saved at {self.config.preprocessor_path}"
            )

        except Exception as e:
            raise CustomException(e, sys)
        
    def _save_transformed_data(self, X_train_tfidf, X_test_tfidf):

        try:

            save_npz(
                self.config.train_data_path,
                X_train_tfidf
            )

            save_npz(
                self.config.test_data_path,
                X_test_tfidf
            )

            logger.info("Transformed train and test data saved successfully.")

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self):

        try:

            logger.info("Starting data transformation...")

            # Read dataset
            df = pd.read_csv(self.config.data_file_path)

            # Create combined text feature
            logger.info("Creating text feature...")

            df = self.preprocessor.create_text_feature(df)

            # Clean text
            logger.info("Cleaning text...")

            df["text"] = df["text"].apply(
                self.preprocessor.clean_text
            )

            # Split data
            logger.info("Splitting data...")
            X_train, X_test, y_train, y_test = self._split_data(df)

            # TF-IDF
            X_train_tfidf, X_test_tfidf, vectorizer = self._fit_vectorizer(
                X_train,
                X_test
            )

            # Save transformed data
            self._save_transformed_data(
                X_train_tfidf,
                X_test_tfidf
            )

            # Save vectorizer
            self._save_preprocessor(vectorizer)

            logger.info("Data transformation completed successfully.")

            return DataTransformationArtifact(

                train_data_path=self.config.train_data_path,

                test_data_path=self.config.test_data_path,

                preprocessor_path=self.config.preprocessor_path,

                X_train=X_train_tfidf,

                X_test=X_test_tfidf,

                y_train=y_train,

                y_test=y_test
            )

        except Exception as e:
            raise CustomException(e, sys)
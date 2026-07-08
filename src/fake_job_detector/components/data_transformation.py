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


class DataTransformation:

    def __init__(self, config: DataTransformationConfig):

        self.config = config

        self.stop_words = set(stopwords.words("english"))

        self.lemmatizer = WordNetLemmatizer()

    def get_wordnet_pos(self, treebank_tag):

        if treebank_tag.startswith("J"):
            return "a"

        elif treebank_tag.startswith("V"):
            return "v"

        elif treebank_tag.startswith("N"):
            return "n"

        elif treebank_tag.startswith("R"):
            return "r"

        return "n"
    
    def _create_text_feature(self, df):

        text_columns = self.config.text_columns

        df[text_columns] = df[text_columns].fillna("")

        df["text"] = df[text_columns].agg(" ".join, axis=1)

        return df
    
    def _clean_text(self, text):

        # Handle missing values
        if pd.isna(text):
            return ""

        # Lowercase
        text = text.lower()

        # Replace non-breaking spaces
        text = text.replace("\xa0", " ")

        # Remove HTML tags
        text = BeautifulSoup(text, "html.parser").get_text()

        # Normalize unicode
        text = unicodedata.normalize("NFKD", text)
        text = text.encode("ascii", "ignore").decode("utf-8")

        # Remove URLs
        text = re.sub(r"http\S+|www\S+|#url_\w+#", " ", text)

        # Remove email addresses
        text = re.sub(r"\S+@\S+", " ", text)

        # Remove MS Office artifacts
        text = re.sub(r"\bmso\b", " ", text)
        text = re.sub(r"\b\d*pt\b", " ", text)
        text = re.sub(r"\b\d*in\b", " ", text)

        # Remove standalone numbers
        text = re.sub(r"\b\d+\b", " ", text)

        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Tokenization
        words = word_tokenize(text)

        # POS tagging
        tagged_words = pos_tag(words)

        # Lemmatization + Stopword Removal
        cleaned_words = [
            self.lemmatizer.lemmatize(
                word,
                self.get_wordnet_pos(tag)
            )
            for word, tag in tagged_words
            if word not in self.stop_words and len(word) > 1
        ]

        return " ".join(cleaned_words)

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

            joblib.dump(
                vectorizer,
                self.config.preprocessor_path
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

            df = self._create_text_feature(df)

            # Clean text
            logger.info("Cleaning text...")
            df["text"] = df["text"].apply(self._clean_text)

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

            return (
                X_train_tfidf,
                X_test_tfidf,
                y_train,
                y_test
            )

        except Exception as e:
            raise CustomException(e, sys)
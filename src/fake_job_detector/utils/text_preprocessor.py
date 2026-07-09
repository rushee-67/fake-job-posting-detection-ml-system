import re
import string
import unicodedata
import pandas as pd

from bs4 import BeautifulSoup

from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class TextPreprocessor:

    def __init__(self, text_columns):

        self.text_columns = text_columns

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
    
    def create_text_feature(self, df):

        df[self.text_columns] = df[self.text_columns].fillna("")

        df["text"] = df[self.text_columns].agg(" ".join, axis=1)

        return df
    
    def create_text(
        self,
        title,
        company_profile,
        description,
        requirements,
        benefits
    ):

        text = " ".join([
            title or "",
            company_profile or "",
            description or "",
            requirements or "",
            benefits or ""
        ])

        return text

    def clean_text(self, text):

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
import re
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

import pandas as pd
import spacy
from tqdm import tqdm

from NLP_Functions.Stopwords_Loader import stop_words

# Load NLP model once with disabled components for speed
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


class Preprocessing:
    def __init__(self, text: str):
        """Initialize with raw text."""
        self.text = text

    def clean_text(self):
        """Removes HTML tags, URLs, mentions, special characters, repeated characters, and extra spaces."""
        self.text = re.sub(r"<.*?>", "", self.text)  # Remove HTML
        self.text = re.sub(r"http\S+|www\.\S+", "", self.text)  # Remove URLs
        self.text = re.sub(r"@\w+", "", self.text)  # Remove mentions
        self.text = re.sub(r"[^a-zA-Z0-9\s]", "", self.text)  # Remove special chars
        self.text = re.sub(r"(.)\1{2,}", r"\1\1", self.text)  # Normalize repeated chars
        self.text = (
            re.sub(r"\s+", " ", self.text).strip().lower()
        )  # Remove extra spaces & lowercase
        return self

    def remove_stopwords(self):
        """Removes stopwords."""
        self.text = " ".join(
            [word for word in self.text.split() if word not in stop_words]
        )
        return self

    def lemmatize_text(self):
        """Lemmatizes the text to its base form."""
        doc = nlp(self.text)
        self.text = " ".join([token.lemma_ for token in doc])
        return self

    def remove_noise_tokens(self):
        """Removes tokens that contain numbers or are shorter than 3 characters."""
        tokens = self.text.split()
        filtered_tokens = [
            token for token in tokens if token.isalpha() and len(token) > 2
        ]
        self.text = " ".join(filtered_tokens)
        return self

    def get_text(self) -> str:
        """Retrieves the final processed text."""
        return self.text


class TextProcessingPipeline:
    def __init__(self, df: pd.DataFrame, text_column: str):
        """
        Initializes the text processing pipeline.

        Args:
            df (pd.DataFrame): The dataframe containing text data.
            text_column (str): The column name with the text to be processed.
        """
        self.df = df
        self.text_column = text_column
        tqdm.pandas()  # Enable progress tracking

    def process(
        self, clean=True, remove_stopwords=True, lemmatize=True, remove_noise=True
    ) -> pd.DataFrame:
        """
        Processes the text data through various preprocessing steps.

        Args:
            clean (bool): Whether to clean the text. Defaults to True.
            remove_stopwords (bool): Whether to remove stopwords. Defaults to True.
            lemmatize (bool): Whether to lemmatize the text. Defaults to True.
            remove_noise (bool): Whether to remove noise (unwanted tokens). Defaults to True.

        Returns:
            pd.DataFrame: The dataframe with an additional column for processed text.
        """
        processed_column = f"processed_{self.text_column}"
        self.df[processed_column] = self.df[self.text_column].astype(str)

        # Use multiprocessing for faster processing
        num_workers = min(cpu_count(), 4)  # Use up to 4 CPU cores
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            self.df[processed_column] = list(
                tqdm(
                    executor.map(
                        self._process_text,
                        self.df[processed_column],
                        [clean] * len(self.df),
                        [remove_stopwords] * len(self.df),
                        [lemmatize] * len(self.df),
                        [remove_noise] * len(self.df),
                    ),
                    total=len(self.df),
                    desc="Processing Text",
                )
            )

        # Remove empty processed texts
        self.df["review_length"] = self.df[processed_column].str.split().str.len()
        self.df = self.df[self.df["review_length"] > 0].reset_index(drop=True)
        return self.df

    def _process_text(
        self, text: str, clean, remove_stopwords, lemmatize, remove_noise
    ) -> str:
        """Applies text preprocessing steps conditionally."""
        processor = Preprocessing(text)
        if clean:
            processor.clean_text()
        if remove_stopwords:
            processor.remove_stopwords()
        if lemmatize:
            processor.lemmatize_text()
        if remove_noise:
            processor.remove_noise_tokens()
        return processor.get_text()

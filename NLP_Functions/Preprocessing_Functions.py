import re

import pandas as pd
import spacy

from NLP_Functions.Stopwords_Loader import stop_words

# Load NLP model
nlp = spacy.load("en_core_web_sm")


class Preprocessing:
    def __init__(self, text: str):
        """
        Initialize with raw text.

        Args:
            text (str): The input text to be processed.
        """
        self.text = text

    def clean_text(self):
        """
        Cleans the text by removing HTML tags, special characters, extra spaces, and converting it to lowercase.

        Returns:
            Preprocessing: The updated instance with cleaned text.
        """
        self.text = re.sub(r"<.*?>", "", self.text)  # Remove HTML tags
        self.text = re.sub(
            r"[^a-zA-Z0-9\s]", "", self.text
        )  # Remove special characters
        self.text = re.sub(r"\s+", " ", self.text).strip()  # Remove extra spaces
        self.text = self.text.lower()

        return self

    def remove_stopwords(self):
        """
        Removes stopwords from the text.

        Returns:
            Preprocessing: The updated instance with stopwords removed.
        """
        self.text = " ".join(
            [word for word in self.text.split() if word not in stop_words]
        )

        return self

    def lemmatize_text(self):
        """
        Lemmatizes the input text by converting words to their base forms.

        Returns:
            Preprocessing: The updated instance with lemmatized text.
        """
        doc = nlp(self.text)
        self.text = " ".join([token.lemma_ for token in doc])

        return self

    def get_text(self) -> str:
        """
        Retrieves the final processed text.

        Returns:
            str: The processed text after all applied transformations.
        """
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

    def process(
        self, clean: bool = True, lemmatize: bool = True, remove_stopwords: bool = True
    ) -> pd.DataFrame:
        """
        Processes the text data through various preprocessing steps.

        Args:
            clean (bool, optional): Whether to clean the text. Defaults to True.
            lemmatize (bool, optional): Whether to lemmatize the text. Defaults to True.
            remove_stopwords (bool, optional): Whether to remove stopwords. Defaults to True.

        Returns:
            pd.DataFrame: The dataframe with an additional column containing processed text.
        """
        processed_column = f"processed_{self.text_column}"
        self.df[processed_column] = (
            self.df[self.text_column]
            .astype(str)
            .apply(
                lambda text: self._process_text(
                    text, clean, remove_stopwords, lemmatize
                )
            )
        )

        return self.df

    def _process_text(
        self, text: str, clean: bool, lemmatize: bool, remove_stopwords: bool
    ) -> str:
        """
        Applies text preprocessing steps conditionally.

        Args:
            text (str): The input text to be processed.
            clean (bool): Whether to clean the text.
            lemmatize (bool): Whether to lemmatize the text.
            remove_stopwords (bool): Whether to remove stopwords.

        Returns:
            str: The fully processed text.
        """
        processor = Preprocessing(text)
        if clean:
            processor.clean_text()
        if remove_stopwords:
            processor.remove_stopwords()
        if lemmatize:
            processor.lemmatize_text()

        return processor.get_text()

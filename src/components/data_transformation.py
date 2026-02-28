from src.logger import logging 
from src.exception import CustomException
from src.utils import clean_text
from src.utils import save_object
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import sys


@dataclass
class DataTransformationConfig:
    tfidf_path: str = "artifacts/tfidf_vectorizer.pkl"
    type_encoder_path: str = "artifacts/ticket_type_encoder.pkl"
    priority_encoder_path: str = "artifacts/ticket_priority_encoder.pkl"


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def data_transformation(self, train_path, test_path):
        try:
            logging.info("Reading training and testing data")

            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)

            # Drop index column safely
            for df in [df_train, df_test]:
                if "Unnamed: 0" in df.columns:
                    df.drop("Unnamed: 0", axis=1, inplace=True)

            logging.info("Removed unnecessary columns")

            # Keep only required columns
            required_cols = ["text", "Ticket Type", "Ticket Priority"]
            df_train = df_train[required_cols]
            df_test = df_test[required_cols]

            # Remove missing target rows
            df_train.dropna(subset=["Ticket Type", "Ticket Priority"], inplace=True)
            df_test.dropna(subset=["Ticket Type", "Ticket Priority"], inplace=True)

            # Fill missing text
            df_train["text"] = df_train["text"].fillna("")
            df_test["text"] = df_test["text"].fillna("")

            # Remove duplicates
            df_train.drop_duplicates(subset=["text"], inplace=True)

            logging.info("Data cleaning completed")

            # Label Encoding
            le_type = LabelEncoder()
            le_priority = LabelEncoder()

            y_train_type = le_type.fit_transform(df_train["Ticket Type"])
            y_test_type = le_type.transform(df_test["Ticket Type"])

            y_train_priority = le_priority.fit_transform(df_train["Ticket Priority"])
            y_test_priority = le_priority.transform(df_test["Ticket Priority"])

            logging.info("Label encoding completed")

            # Clean text
            df_train["text"] = df_train["text"].apply(clean_text)
            df_test["text"] = df_test["text"].apply(clean_text)

            # TF-IDF
            tfidf = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words="english"
            )

            X_train = tfidf.fit_transform(df_train["text"]).toarray()
            X_test = tfidf.transform(df_test["text"]).toarray()

            logging.info("TF-IDF transformation completed")

            # Combine features and targets
            train_arr = np.c_[X_train, y_train_type, y_train_priority]
            test_arr = np.c_[X_test, y_test_type, y_test_priority]

            # Save objects
            save_object(self.data_transformation_config.tfidf_path, tfidf)
            save_object(self.data_transformation_config.type_encoder_path, le_type)
            save_object(self.data_transformation_config.priority_encoder_path, le_priority)

            logging.info("Transformation artifacts saved")

            return train_arr, test_arr

        except Exception as e:
            raise CustomException(e, sys)
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
        self.data_transformation_config=DataTransformationConfig()
 
    def data_transformation(self, train_path,test_path):
        try:
            logging.info("Reading the data")
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)
            logging.info("Reading the data is complete")

            df_train.drop("Unnamed: 0",axis=1,inplace=True)
            df_test.drop("Unnamed: 0",axis=1,inplace=True)
            logging.info("Droping useless column")

            le1= LabelEncoder()
            le2= LabelEncoder()

            y_train_type=le1.fit_transform(df_train['Ticket Type'])
            y_test_type=le1.transform(df_test['Ticket Type'])

            y_train_priority=le2.fit_transform(df_train['Ticket Priority'])
            y_test_priority=le2.transform(df_test['Ticket Priority'])
            
            logging.info("Fucckkk label encoding is done")

            df_train['text']=df_train['text'].apply(clean_text)
            df_test['text']=df_test['text'].apply(clean_text)

            tfidf=TfidfVectorizer(max_features=5000,ngram_range=(1,2),stop_words='english')
            X_train=tfidf.fit_transform(df_train['text']).toarray()
            X_test=tfidf.transform(df_test['text']).toarray()

            logging.info("tfidf is doneee")

            train_arr = np.c_[
                X_train,
                y_train_type,
                y_train_priority
            ]

            test_arr = np.c_[
                X_test,
                y_test_type,
                y_test_priority
            ]

            save_object(
                file_path=self.data_transformation_config.tfidf_path,
                obj=tfidf
            )

            save_object(
                file_path=self.data_transformation_config.type_encoder_path,
                obj=le1
            )

            save_object(
                file_path=self.data_transformation_config.priority_encoder_path,
                obj=le2
            )

            return(train_arr,
                   test_arr,
                   )

        except Exception as e:
            raise CustomException(e, sys)
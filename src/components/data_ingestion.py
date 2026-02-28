import os 
import sys

import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException

@dataclass
class DataIngestionConfig:
    train_data_path :str= os.path.join("artifacts",'train.csv')
    test_data_path :str= os.path.join("artifacts",'test.csv')

class DataIngestion:
    def __init__(self):    
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Reading the data")
            df= pd.read_csv('notebook/data/processed_data.csv')
            os.makedirs('artifacts',exist_ok=True)
            logging.info("Starting train test split")
            df_train,df_test=train_test_split(df,train_size=0.80,random_state=47)

            df_train.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            df_test.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Split and ingestion is completed")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)




    
 
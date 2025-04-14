import os
import sys
import pandas as pd
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from src.mlproject.logger import logging
from src.mlproject.components.mongo import MongoDBClient
from src.mlproject.exception import CustomException
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")

class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()  # FIXED: unified naming
        self.mongo_client = MongoDBClient()

    def initiate_data_ingestion(self):
        logging.info("Exporting data from MongoDB")
        try:
            df = self.mongo_client.export_collection_to_dataframe("students")
            logging.info("Data exported from MongoDB")

            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)

            df.to_csv(self.config.raw_data_path, index=False)
            logging.info(f"Raw data saved at: {self.config.raw_data_path}")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.config.train_data_path, index=False, header=True)
            test_set.to_csv(self.config.test_data_path, index=False, header=True)

            logging.info("Train and test split completed.")
            logging.info(f"Train path: {self.config.train_data_path}, Test path: {self.config.test_data_path}")

            return (
                self.config.train_data_path,
                self.config.test_data_path
            )

        except Exception as e:
            logging.error("Error occurred in data ingestion stage")
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()

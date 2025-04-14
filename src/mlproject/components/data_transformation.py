import os
import sys
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import joblib

from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessing.pkl")

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
    
    def get_data_transormation_object(self):
        try:
            logging.info("Data Transformation Started")

            numerical_columns = ["reading_score","writing_score"]
            categorical_columns = ["gender","race_ethnicity","parental_level_of_education", 'lunch', 'test_preparation_course']

            num_pipeline = Pipeline(steps = [
                ("imputer",SimpleImputer(strategy = "mean")),
                ("scaler",StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ("imputer",SimpleImputer(strategy = "most_frequent")),
                ("onehot",OneHotEncoder(handle_unknown="ignore")),
                ("scaler",StandardScaler(with_mean = False))
            ])

            preprocessor = ColumnTransformer(transformers=[
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)

            logging.info("Read the train and test data")

            target_column = "math_score"
            drop_columns = [target_column]

            input_feature_train_df = df_train.drop(columns = drop_columns,axis = 1)
            target_feature_train_df = df_train[target_column]

            input_feature_test_df = df_test.drop(columns=drop_columns, axis=1)
            target_feature_test_df = df_test[target_column]

            preprocessor = self.get_data_transormation_object()

            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            # Save the Preprocessor
            joblib.dump(preprocessor,self.config.preprocessor_obj_file_path)

            logging.info("Preprocessing complete and object saved")

            return (
                input_feature_train_arr,
                target_feature_train_df,
                input_feature_test_arr,
                target_feature_test_df,
                self.config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)
if __name__ == "__main__":
    from src.mlproject.components.data_ingestion import DataIngestion

    data_ingestion = DataIngestion()
    train_path,test_path = data_ingestion.initiate_data_ingestion()

    data_transformation=DataTransformation()
    data_transformation.initiate_data_transformation(train_path,test_path)
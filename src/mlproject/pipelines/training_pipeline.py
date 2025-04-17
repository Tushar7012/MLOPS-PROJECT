from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_transformation import DataTransformation
from src.mlproject.components.model_trainer import ModelTrainer
import numpy as np

def start_training_pipeline():

    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    transformation = DataTransformation()
    X_train, X_test, y_train, y_test = transformation.initiate_data_transformation(train_path, test_path)

    # Concatenate X and y to match model_trainer's expectations
    train_arr = np.c_[X_train, y_train]
    test_arr = np.c_[X_test, y_test]

    trainer = ModelTrainer()
    trainer.initiate_model_trainer(train_arr, test_arr)

if __name__ == "__main__":
    start_training_pipeline()

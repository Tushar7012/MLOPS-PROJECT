import pandas as pd
from src.mlproject.utils import load_object
from src.mlproject.exception import CustomException
import numpy as np
import os

class PredictPipeline:
    def __init__(self):
        self.model_path = "artifacts/model.pkl"
        self.preprocessor_path = "artifacts/preprocessor.pkl"

    def load_model(self):
        try:
            # Load the model from the saved file
            model = load_object(self.model_path)
            return model
        except Exception as e:
            raise CustomException(f"Error in loading model: {str(e)}")

    def load_preprocessor(self):
        try:
            # Load the preprocessor from the saved file
            preprocessor = load_object(self.preprocessor_path)
            return preprocessor
        except Exception as e:
            raise CustomException(f"Error in loading preprocessor: {str(e)}")

    def predict(self, input_data):
        try:
            # Load the preprocessor and model
            model = self.load_model()
            preprocessor = self.load_preprocessor()

            # Apply transformations to the input data
            input_data_transformed = preprocessor.transform(input_data)

            # Predict using the trained model
            predictions = model.predict(input_data_transformed)
            return predictions
        except Exception as e:
            raise CustomException(f"Error in prediction: {str(e)}")

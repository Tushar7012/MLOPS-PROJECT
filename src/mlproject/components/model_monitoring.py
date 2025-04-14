import pandas as pd
import numpy as np
import os
import sys
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.utils import load_object
from sklearn.metrics import accuracy_score
from scipy.stats import ks_2samp

class ModelMonitoring:
    def __init__(self, model_path, preprocessor_path, reference_data_path):
        self.model = load_object(model_path)
        self.preprocessor = load_object(preprocessor_path)
        self.reference_data = pd.read_csv(reference_data_path)

    def check_prediction_drift(self, current_data_path):
        try:
            logging.info("Checking prediction drift...")

            current_data = pd.read_csv(current_data_path)
            reference_features = self.reference_data.drop("math score", axis=1)
            current_features = current_data.drop("math score", axis=1)

            # Transform
            reference_transformed = self.preprocessor.transform(reference_features)
            current_transformed = self.preprocessor.transform(current_features)

            # Predict
            ref_preds = self.model.predict(reference_transformed)
            cur_preds = self.model.predict(current_transformed)

            # Drift test (KS test)
            drift_stat, drift_pval = ks_2samp(ref_preds, cur_preds)

            logging.info(f"Drift p-value: {drift_pval}")

            if drift_pval < 0.05:
                logging.warning("Prediction drift detected! Model performance may be degrading.")
                return False
            else:
                logging.info("No prediction drift detected.")
                return True

        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    monitor = ModelMonitoring(
        model_path=os.path.join("artifacts", "model.pkl"),
        preprocessor_path=os.path.join("artifacts", "preprocessor.pkl"),
        reference_data_path=os.path.join("artifacts", "train.csv")
    )
    monitor.check_prediction_drift(current_data_path=os.join.path("artifacts","test.csv"))
    
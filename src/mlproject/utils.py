import os
import sys
import dill
import pandas as pd
from src.mlproject.exception import CustomException

def save_object(file_path,obj):
    """
    Save any Python Object(lie model or preprocessor)
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok = True)

        with open(file_path,"wb") as f:
            dill.dump(obj,f)
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,"rb") as f:
            return dill.load(f)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(true,predicted):
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    r2 = r2_score(true, predicted)
    mse = mean_squared_error(true, predicted)
    mae = mean_absolute_error(true, predicted)

    return {
        "R2 Score": r2,
        "MSE": mse,
        "MAE": mae
    }
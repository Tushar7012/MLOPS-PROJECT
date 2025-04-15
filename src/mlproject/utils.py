import os
import sys
import dill
import pandas as pd
from pymongo import MongoClient
from src.mlproject.logger import logging
from dotenv import load_dotenv
from src.mlproject.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import pickle

def read_mongo_data():
    logging.info("Reading MongoDB database started")
    
    try:
        # Load environment variables
        load_dotenv()

        mongo_uri = os.getenv("MONGO_DB_URI")
        db_name = os.getenv("MONGO_DB_NAME")
        collection_name = os.getenv("COLLECTION_NAME")

        # Connect to MongoDB
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]

        # Read data from MongoDB and convert to DataFrame
        data = list(collection.find({}))
        df = pd.DataFrame(data)

        # Optional: Drop the MongoDB _id field
        if "_id" in df.columns:
            df.drop(columns=["_id"], inplace=True)

        logging.info("Connection Established and Data Read from MongoDB")
        print(df.head())

        return df

    except Exception as ex:
        raise CustomException(ex)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path,"rb") as f:
            return dill.load(f)
    except Exception as e:
        raise CustomException(e,sys)

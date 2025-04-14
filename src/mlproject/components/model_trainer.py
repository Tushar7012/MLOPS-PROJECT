import sys
import json
import os
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import GridSearchCV

from src.mlproject.utils import save_object
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging


@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join("artifacts", "model.pkl")
    report_path: str = os.path.join("artifacts", "model_report.json")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array, task_type="regression"):
        try:
            logging.info(f"Task Type: {task_type}")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            if task_type == "regression":
                models = {
                    "LinearRegression": LinearRegression(),
                    "RandomForestRegressor": RandomForestRegressor()
                }
                param_grid = {
                    "RandomForestRegressor": {
                        "n_estimators": [50, 100],
                        "max_depth": [None, 10]
                    }
                }
                scoring = "r2"
                evaluate = r2_score

            elif task_type == "classification":
                models = {
                    "LogisticRegression": LogisticRegression(max_iter=1000),
                    "RandomForestClassifier": RandomForestClassifier()
                }
                param_grid = {
                    "RandomForestClassifier": {
                        "n_estimators": [50, 100],
                        "max_depth": [None, 10]
                    }
                }
                scoring = "accuracy"
                evaluate = accuracy_score

            else:
                raise ValueError("Invalid task_type. Choose 'regression' or 'classification'.")

            best_model = None
            best_score = -1
            best_model_name = ""
            report = {}

            for name, model in models.items():
                if name in param_grid:
                    logging.info(f"Tuning hyperparameters for {name}")
                    grid = GridSearchCV(model, param_grid[name], scoring=scoring, cv=3)
                    grid.fit(X_train, y_train)
                    model = grid.best_estimator_
                else:
                    model.fit(X_train, y_train)

                preds = model.predict(X_test)
                score = evaluate(y_test, preds)

                report[name] = score
                logging.info(f"{name} {scoring}: {score}")

                if score > best_score:
                    best_model = model
                    best_score = score
                    best_model_name = name

            save_object(self.config.model_path, best_model)
            logging.info(f"Saved best model: {best_model_name} to {self.config.model_path}")

            with open(self.config.report_path, "w") as f:
                json.dump(report, f, indent=4)

            logging.info(f"Best Model: {best_model_name} with {scoring}: {best_score}")
            return best_model_name, best_score

        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    import sys
    from src.mlproject.components.data_transformation import DataTransformation
    from src.mlproject.components.model_trainer import ModelTrainer

    # Paths from DataIngestion (assuming you already ingested and transformed the data)
    train_path = "artifacts/train.csv"
    test_path = "artifacts/test.csv"

    try:
        # Step 1: Data Transformation
        transformer = DataTransformation()
        train_arr, test_arr = transformer.initiate_data_transformation(train_path, test_path)

        # Step 2: Model Training
        trainer = ModelTrainer()
        model_score = trainer.initiate_model_training(train_arr, test_arr)
        print(f"\n Model trained successfully. Best model score: {model_score}\n")

    except Exception as e:
        print(f"\n Something went wrong: {e}")
        sys.exit(1)

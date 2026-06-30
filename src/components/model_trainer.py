import os
import sys
import numpy as np
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils.utils import save_object


# ---------------------------
# CONFIG CLASS
# ---------------------------
@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join(
        "artifacts",
        "model.pkl"
    )


# ---------------------------
# MODEL TRAINER CLASS
# ---------------------------
class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def train_model(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("Model training started")

            # Models
            models = {
                "LinearRegression": LinearRegression(),
                "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42)
            }

            best_model = None
            best_score = -float("inf")

            # Train + Evaluate
            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                score = r2_score(y_test, preds)

                logging.info(f"{name} R2 Score: {score}")

                if score > best_score:
                    best_score = score
                    best_model = model

            if best_model is None:
                raise CustomException("No best model found")

            logging.info(f"Best Model selected with R2 Score: {best_score}")

            # Save model
            save_object(
                file_path=self.config.trained_model_file_path,
                obj=best_model
            )

            logging.info(f"Model saved at {self.config.trained_model_file_path}")

            return best_score

        except Exception as e:
            raise CustomException(e, sys)
import sys
import os
import numpy as np
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object


@dataclass
class ModelTrainerConfig:
    ticket_type_model_path: str = os.path.join("artifacts", "ticket_type_model.pkl")
    ticket_priority_model_path: str = os.path.join("artifacts", "ticket_priority_model.pkl")


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):

        try:
            logging.info("Model training pipeline started")

            # =========================
            # Split Features and Targets
            # =========================

            X_train = train_arr[:, :-2]
            y_train_type = train_arr[:, -2].astype(int)
            y_train_priority = train_arr[:, -1].astype(int)

            X_test = test_arr[:, :-2]
            y_test_type = test_arr[:, -2].astype(int)
            y_test_priority = test_arr[:, -1].astype(int)

            logging.info("Feature-target splitting completed")

            # =========================
            # Define Models
            # =========================

            models = {
                "Logistic Regression": LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42
                ),
                "Random Forest": RandomForestClassifier(
                    class_weight="balanced",
                    random_state=42
                ),
                "Linear SVC": LinearSVC(),
                "KNN": KNeighborsClassifier()
            }

            param_grids = {
                "Logistic Regression": {
                    "C": [0.01, 0.1, 1, 10]
                },
                "Random Forest": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10, 20]
                },
                "Linear SVC": {
                    "C": [0.01, 0.1, 1, 10]
                },
                "KNN": {
                    "n_neighbors": [3, 5, 7]
                }
            }

            # =========================
            # Train Ticket Type Model
            # =========================

            logging.info("Training model for Ticket Type")

            report_type, best_type_name, best_type_model = evaluate_models(
                X_train,
                y_train_type,
                X_test,
                y_test_type,
                models,
                param_grids
            )

            logging.info(f"Best Ticket Type Model: {best_type_name}")

            # =========================
            # Train Ticket Priority Model
            # =========================

            logging.info("Training model for Ticket Priority")

            report_priority, best_priority_name, best_priority_model = evaluate_models(
                X_train,
                y_train_priority,
                X_test,
                y_test_priority,
                models,
                param_grids
            )

            logging.info(f"Best Ticket Priority Model: {best_priority_name}")

            # =========================
            # Save Best Models
            # =========================

            save_object(
                file_path=self.model_trainer_config.ticket_type_model_path,
                obj=best_type_model
            )

            save_object(
                file_path=self.model_trainer_config.ticket_priority_model_path,
                obj=best_priority_model
            )

            logging.info("Best models saved successfully")

            return (
                report_type,
                report_priority,
                best_type_name,
                best_priority_name
            )

        except Exception as e:
            raise CustomException(e, sys)
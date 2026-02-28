import sys
import os
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import clean_text, load_object


@dataclass
class PredictionPipelineConfig:
    ticket_type_model_path: str = os.path.join("artifacts", "ticket_type_model.pkl")
    ticket_priority_model_path: str = os.path.join("artifacts", "ticket_priority_model.pkl")
    tfidf_path: str = os.path.join("artifacts", "tfidf_vectorizer.pkl")
    type_encoder_path: str = os.path.join("artifacts", "ticket_type_encoder.pkl")
    priority_encoder_path: str = os.path.join("artifacts", "ticket_priority_encoder.pkl")


class PredictionPipeline:

    def __init__(self):
        self.config = PredictionPipelineConfig()

    def predict(self, text_input):

        try:
            logging.info("Prediction pipeline started")

            # =========================
            # Load artifacts
            # =========================

            type_model = load_object(self.config.ticket_type_model_path)
            priority_model = load_object(self.config.ticket_priority_model_path)
            tfidf = load_object(self.config.tfidf_path)
            type_encoder = load_object(self.config.type_encoder_path)
            priority_encoder = load_object(self.config.priority_encoder_path)

            logging.info("Artifacts loaded successfully")

            # =========================
            # Preprocess input
            # =========================

            cleaned_text = clean_text(text_input)

            transformed_text = tfidf.transform([cleaned_text])
            # Keep sparse format (better for text models)

            # =========================
            # Predict
            # =========================

            type_pred = type_model.predict(transformed_text)
            priority_pred = priority_model.predict(transformed_text)

            final_type = type_encoder.inverse_transform(type_pred)[0]
            final_priority = priority_encoder.inverse_transform(priority_pred)[0]

            logging.info("Prediction completed successfully")

            return {
                "Ticket Type": final_type,
                "Ticket Priority": final_priority
            }

        except Exception as e:
            raise CustomException(e, sys)
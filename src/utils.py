import re
import nltk
from nltk.stem import PorterStemmer
ps =PorterStemmer()
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
def clean_text(x):
    x = str(x)
    x = re.sub('[^a-zA-Z]',' ',x)
    x = x.lower()
    x = x.split()
    x = [ps.stem(word) for word in x if word not in stop_words]
    x = ' '.join(x)
    return x


import os
import sys
import pickle
from src.exception import CustomException
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


import os
import sys
import pickle

from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging


def evaluate_models(X_train, y_train, X_test, y_test, models, param_grids):

    try:
        report = {}
        best_model = None
        best_score = 0
        best_model_name = ""

        for model_name in models:

            logging.info(f"Tuning hyperparameters for {model_name}")

            model = models[model_name]
            params = param_grids[model_name]

            gs = GridSearchCV(
                model,
                params,
                cv=5,
                scoring="accuracy",
                n_jobs=-1,
                verbose=1
            )

            gs.fit(X_train, y_train)

            best_trained_model = gs.best_estimator_

            y_test_pred = best_trained_model.predict(X_test)

            from sklearn.metrics import accuracy_score
            test_accuracy = accuracy_score(y_test, y_test_pred)

            report[model_name] = test_accuracy

            logging.info(f"{model_name} accuracy: {test_accuracy}")
            logging.info(f"{model_name} best params: {gs.best_params_}")

            print(f"{model_name} Accuracy: {test_accuracy}")
            print(f"{model_name} Best Params: {gs.best_params_}")
            print("----------------------------------------")

            if test_accuracy > best_score:
                best_score = test_accuracy
                best_model = best_trained_model
                best_model_name = model_name

        print(f"\nBest Model: {best_model_name}")
        print(f"Best Accuracy: {best_score}")

        return report, best_model_name, best_model

    except Exception as e:
        raise CustomException(e, sys)


import pickle
import sys
from src.exception import CustomException


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
import logging
from typing import Dict, Tuple

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score, recall_score


# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import max_error, mean_absolute_error, r2_score
# from sklearn.model_selection import train_test_split
#
#
# def split_data(data: pd.DataFrame, parameters: Dict) -> Tuple:
#     """Splits data into features and targets training and test sets.
#
#     Args:
#         data: Data containing features and target.
#         parameters: Parameters defined in parameters/data_science.yml.
#     Returns:
#         Split data.
#     """
#     X = data[parameters["features"]]
#     y = data["price"]
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
#     )
#     return X_train, X_test, y_train, y_test
#
#
# def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
#     """Trains the linear regression model.
#
#     Args:
#         X_train: Training data of independent features.
#         y_train: Training data for price.
#
#     Returns:
#         Trained model.
#     """
#     regressor = LinearRegression()
#     regressor.fit(X_train, y_train)
#     return regressor
#
#
# def evaluate_model(
#     regressor: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series
# ) -> Dict[str, float]:
#     """Calculates and logs the coefficient of determination.
#
#     Args:
#         regressor: Trained model.
#         X_test: Testing data of independent features.
#         y_test: Testing data for price.
#     """
#     y_pred = regressor.predict(X_test)
#     score = r2_score(y_test, y_pred)
#     mae = mean_absolute_error(y_test, y_pred)
#     me = max_error(y_test, y_pred)
#     logger = logging.getLogger(__name__)
#     logger.info("Model has a coefficient R^2 of %.3f on test data.", score)
#     return {"r2_score": score, "mae": mae, "max_error": me}



def select_hiperparameters(X, Y):
    pipeline = Pipeline([
        ('clf', GradientBoostingClassifier())
    ])

    # Define parameter grid for GridSearchCV
    param_grid = {
        'clf__n_estimators': [50], #100, 150, 200],
        'clf__learning_rate': [0.01],# 0.1, 0.2],
        'clf__max_depth': [3],# 4, 5, 6]
    }

    # Create GridSearchCV object
    grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='f1', n_jobs=-1)

    grid_search.fit(X, Y['is_fraud'].squeeze())
    best_params = {k.replace('clf__', ''): v for k, v in grid_search.best_params_.items()}

    logger = logging.getLogger(__name__)
    logger.info("Model has best parameters :", best_params)

    return _train_final_model(X, Y, best_params)


def _train_final_model(X, Y, best_params) -> GradientBoostingClassifier:
    final_model = GradientBoostingClassifier(**best_params)
    final_model.fit(X, Y['is_fraud'].squeeze())
    return final_model

def _get_metrics(Y_true, Y_pred):
    return {
        'roc_auc': roc_auc_score(Y_true, Y_pred),
        'accuracy_score': balanced_accuracy_score(Y_true, Y_pred),
        'recall_score': recall_score(Y_true, Y_pred),
    }

def evaluate_model(X, Y, model):
    Y_pred = model.predict(X)

    metrics = _get_metrics(Y, Y_pred)

    logger = logging.getLogger(__name__)
    logger.info("Final model score :", metrics)

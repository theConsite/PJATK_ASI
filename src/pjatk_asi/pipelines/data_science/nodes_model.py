import logging
from typing import Dict, Tuple

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score, recall_score


def select_hiperparameters(X: pd.DataFrame, Y: pd.DataFrame) -> GradientBoostingClassifier:
    pipeline = Pipeline([
        ('clf', GradientBoostingClassifier())
    ])
    # Define parameter grid for GridSearchCV
    param_grid = {
        'clf__n_estimators': [50, 100, 150, 200],
        'clf__learning_rate': [0.01, 0.1, 0.2],
        'clf__max_depth': [3, 4, 5, 6]
    }
    # Create GridSearchCV object
    grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X, Y)
    best_params = {k.replace('clf__', ''): v for k, v in grid_search.best_params_.items()}
    logger = logging.getLogger(__name__)
    logger.info("Model has best parameters: %s", str(best_params))

    return _train_final_model(X, Y, best_params)


def _train_final_model(X: pd.DataFrame, Y: pd.DataFrame, best_params: Dict[str, str]) -> GradientBoostingClassifier:
    final_model = GradientBoostingClassifier(**best_params)
    final_model.fit(X, Y)
    return final_model


def _get_metrics(Y_true: pd.DataFrame, Y_pred: pd.DataFrame):
    return {
        'roc_auc': roc_auc_score(Y_true, Y_pred),
        'accuracy_score': balanced_accuracy_score(Y_true, Y_pred),
        'recall_score': recall_score(Y_true, Y_pred),
    }


def evaluate_model(X: pd.DataFrame, Y: pd.DataFrame, model: GradientBoostingClassifier):
    Y_pred = model.predict(X)
    metrics = _get_metrics(Y, Y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Final model score: %s", str(metrics))

    return Y_pred

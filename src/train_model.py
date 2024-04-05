from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pickle


def select_hiperparameters(X, Y):
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
    grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='f1', n_jobs=-1)

    grid_search.fit(X, Y)

    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Score:", grid_search.best_score_)

    return {k.replace('clf__', ''): v for k, v in grid_search.best_params_.items()}

def train_final_model(X, Y, best_params):
    final_model = GradientBoostingClassifier(**best_params)
    final_model.fit(X, Y)

    with open("../bin_store/Model.bin", 'wb') as f:
        pickle.dump(final_model, file=f)


if __name__ == "__main__":
    from data_load import read_data
    from feature_engineering import feature_generation, remove_columns
    from data_preperation import (data_split, train_test_split, balance,
                                  one_hot_encode_fit, one_hot_encode_apply,
                                  normalize_columns_fit, normalize_columns_apply)
    fraud_df = read_data()
    feature_generation(fraud_df)
    remove_columns(fraud_df)
    X, Y = data_split(fraud_df)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    X_train = one_hot_encode_fit(X_train)
    X_train = normalize_columns_fit(X_train)

    X_test = one_hot_encode_apply(X_test)
    X_test = normalize_columns_apply(X_test)

    X_train, Y_train = balance(X_train, Y_train)

    best_params = select_hiperparameters(X_train, Y_train)
    train_final_model(X_train, Y_train, best_params)

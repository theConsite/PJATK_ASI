from data_load import read_data
from feature_engineering import feature_generation, remove_columns
from data_preperation import (data_split, train_test_split, balance,
                              one_hot_encode_fit, one_hot_encode_apply,
                              normalize_columns_fit, normalize_columns_apply)
from train_model import select_hiperparameters, train_final_model
from train_evaluation import evaluate_model

if __name__ == "__main__":
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
    
    evaluate_model(X_test, Y_test)
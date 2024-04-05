from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import pickle
from imblearn.over_sampling import SMOTE

def data_split(fraud_df):
    Y = fraud_df[['is_fraud']]
    X = fraud_df.drop(columns=['is_fraud'])
    return X, Y


def test_train_split(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5, stratify=Y['is_fraud'])
    return X_train, X_test, Y_train, Y_test


def get_cat_columns(X):
    # cechy o typach nie numerycznych
    cat_columns = X.select_dtypes(exclude=["bool_", "number"]).columns.values.tolist()

    # szokamy również danych o charakterze categorycznym zapisanych w kolumnach numerycznych
    for col in X.select_dtypes(include=["number"]).columns:
        c_uniq = len(X[col].unique())
        if c_uniq > 2:
            print(f'{col}: {c_uniq}')

    return cat_columns

def one_hot_encode_apply(X):
    cat_columns = get_cat_columns(X)
    cat_transformer_X = None
    with open("../bin_store/OneHotEncoder.bin", 'rb') as f:
        cat_transformer_X = pickle.load(f)

    ohe_df = pd.DataFrame(cat_transformer_X.transform(X), columns=cat_transformer_X.get_feature_names_out())
    X = pd.concat([X.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1).drop(cat_columns, axis=1)

    return X

def one_hot_encode_fit(X):
    cat_columns = get_cat_columns(X)
    cat_transformer_X = make_column_transformer(
        (OneHotEncoder(sparse_output=False), cat_columns),
        sparse_threshold=0,
        verbose_feature_names_out=False,
        remainder='drop')

    ohe_df = pd.DataFrame(cat_transformer_X.fit_transform(X), columns=cat_transformer_X.get_feature_names_out())
    X = pd.concat([X.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1).drop(cat_columns, axis=1)

    with open("../bin_store/OneHotEncoder.bin", 'wb') as f:
        pickle.dump(cat_transformer_X, file=f)

    return X


def normalize_columns_apply(X):
    quant_columns = X.select_dtypes(include=["number"]).columns.values.tolist()
    std_scaler = None
    with open("../bin_store/StdScaler.bin", 'rb') as f:
        std_scaler = pickle.load(file=f)

    X[quant_columns] = std_scaler.transform(X[quant_columns])

    return X


def normalize_columns_fit(X):
    quant_columns = X.select_dtypes(include=["number"]).columns.values.tolist()
    std_scaler = StandardScaler()
    X[quant_columns] = std_scaler.fit_transform(X[quant_columns])

    with open("../bin_store/StdScaler.bin", 'wb') as f:
        pickle.dump(std_scaler, file=f)

    return X


def balance(X, Y):
    smote = SMOTE(sampling_strategy='auto') #TODO: change to SMOTENC, then change order to normalize -> balance -> ohe

    n = 3000
    X.reset_index(drop=True, inplace=True)
    Y.reset_index(drop=True, inplace=True)

    Y_under_sampled = Y.groupby('is_fraud', group_keys=False).apply(lambda x: x.sample(min(len(x), n), replace=False))
    X_under_sampled = X.loc[Y_under_sampled.index]

    X, Y = smote.fit_resample(X_under_sampled, Y_under_sampled['is_fraud'])

    return X, Y


if __name__ == "__main__":
    from data_load import read_data
    from feature_engineering import feature_generation, remove_columns
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

    print(X_train.describe())
    print(X_test.describe())
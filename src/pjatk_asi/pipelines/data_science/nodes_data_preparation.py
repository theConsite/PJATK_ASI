from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
from imblearn.over_sampling import SMOTE


def data_split(fraud_df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    Y = fraud_df[['is_fraud']]
    X = fraud_df.drop(columns=['is_fraud'])
    return X, Y


def test_train_split(X: pd.DataFrame, Y: pd.DataFrame):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5, stratify=Y['is_fraud'])
    return X_train, X_test, Y_train, Y_test


def _get_cat_columns(X: pd.DataFrame) -> pd.DataFrame:
    # cechy o typach nie numerycznych
    cat_columns = X.select_dtypes(exclude=["bool_", "number"]).columns.values.tolist()

    # szokamy również danych o charakterze categorycznym zapisanych w kolumnach numerycznych
    #for col in X.select_dtypes(include=["number"]).columns:
        #c_uniq = len(X[col].unique())
        #if c_uniq > 2:
            #print(f'{col}: {c_uniq}')

    return cat_columns


def one_hot_encode_apply(X: pd.DataFrame, transformer: ColumnTransformer) -> pd.DataFrame:
    cat_columns = _get_cat_columns(X)
    ohe_df = pd.DataFrame(transformer.transform(X), columns=transformer.get_feature_names_out())
    X = pd.concat([X.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1).drop(cat_columns, axis=1)

    return X


def one_hot_encode_fit(X: pd.DataFrame) -> (pd.DataFrame, ColumnTransformer):
    cat_columns = _get_cat_columns(X)
    cat_transformer_X = make_column_transformer(
        (OneHotEncoder(sparse_output=False), cat_columns),
        sparse_threshold=0,
        verbose_feature_names_out=False,
        remainder='drop')

    ohe_df = pd.DataFrame(cat_transformer_X.fit_transform(X), columns=cat_transformer_X.get_feature_names_out())
    X = pd.concat([X.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1).drop(cat_columns, axis=1)

    return X, cat_transformer_X


# def normalize_columns_apply(X: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
#     quant_columns = X.select_dtypes(include=["number"]).columns.values.tolist()
#     X[quant_columns] = scaler.transform(X[quant_columns])
#
#     return X


def normalize_columns_apply(X: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """
    Normalize only the columns that were seen by the scaler and remove columns
    that were not part of the scaler's input.

    Parameters:
    X (pd.DataFrame): The DataFrame with the features to be normalized.
    scaler (StandardScaler): The fitted StandardScaler object.

    Returns:
    pd.DataFrame: The DataFrame with the normalized columns and only the columns
                  that were part of the scaler's input.
    """
    # Get the columns that were seen by the scaler (the scaler's feature names)
    seen_columns = scaler.feature_names_in_

    # Find common columns between the DataFrame and the scaler's feature names
    common_columns = [col for col in seen_columns if col in X.columns]

    if not common_columns:
        raise ValueError("No common columns between the DataFrame and the scaler.")

    # Normalize the columns that were seen by the scaler
    X[common_columns] = scaler.transform(X[common_columns])

    # Keep only the columns that were seen by the scaler
    X = X[common_columns]

    return X

def normalize_columns_fit(X: pd.DataFrame) -> (pd.DataFrame, StandardScaler):
    quant_columns = X.select_dtypes(include=["number"]).columns.values.tolist()
    std_scaler = StandardScaler()
    X[quant_columns] = std_scaler.fit_transform(X[quant_columns])
    return X, std_scaler


def balance(X: pd.DataFrame, Y: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    smote = SMOTE(sampling_strategy='auto') #TODO: change to SMOTENC, then change order to normalize -> balance -> ohe

    n = 3000
    X.reset_index(drop=True, inplace=True)
    Y.reset_index(drop=True, inplace=True)

    Y_under_sampled = Y.groupby('is_fraud', group_keys=False).apply(lambda x: x.sample(min(len(x), n), replace=False))
    X_under_sampled = X.loc[Y_under_sampled.index]

    X_oversampled, Y_oversampled = smote.fit_resample(X_under_sampled, Y_under_sampled['is_fraud'])

    return X_oversampled, Y_oversampled
import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier
import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir + "\\pipelines\\data_processing")
sys.path.append(parent_dir + "\\pipelines\\data_science")
import nodes_feature_engineering as nfe
import nodes_data_preparation as ndp
print(parent_dir + "\\pipelines\\data_processing")
#########################################################
#req model.pickle in web_app catalog
#########################################################
model = None
with open(parent_dir + '\\web_app\\model.pickle', 'rb') as file:
    model: GradientBoostingClassifier = pickle.load(file)

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    df = nfe.feature_generation(df.copy())
    df = nfe.remove_columns(df.copy())
    X_train_ohe,one_hot_encoder = ndp.one_hot_encode_fit(df.copy())
    X_train_prepared,std_scaler = ndp.normalize_columns_fit(X_train_ohe.copy())
    X_test_ohe = ndp.one_hot_encode_apply(df.copy(),one_hot_encoder)
    X_test_prepared = ndp.normalize_columns_apply(X_test_ohe.copy(),std_scaler)
    return X_test_prepared


st.title("Upload CSV and Predict")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.write(data)

    if st.button('Predict'):
        transformed_data = transform_data(data)
        predictions = model.predict(transformed_data)
        #fe956c7e4a253c437c18918bf96f7b62

        data['is_fraud'] = predictions
        final = data
        st.write(final)

        prediction_counts = pd.Series(predictions).value_counts()
        st.write("Prediction Counts:")
        st.write(prediction_counts)

        csv = final.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download predictions as CSV",
            data=csv,
            file_name='predictions.csv',
            mime='text/csv',
        )

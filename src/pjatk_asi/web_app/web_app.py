import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier

print("Start")
model = None
with open('C:\\Users\\filip\\PycharmProjects\\PJATK_ASI\\data\\06_models\\model.pickle\\2024-05-31T20.05.39.596Z\\model.pickle', 'rb') as file:
    model:GradientBoostingClassifier = pickle.load(file)



st.title("Upload CSV")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data)
    data.to_csv('data/01_raw/dataset.csv', index=False)

    model.predict(data)

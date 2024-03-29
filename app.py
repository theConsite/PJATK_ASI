import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression

file= 'dataset.csv'
fraud_data = pd.read_csv(file)

#Usuwamy kolumne z liczba porzadkowa
fraud_data.drop('Unnamed: 0', axis=1, inplace=True)

#Tu printujemy analizę danych, przyda się przy opracowywaniu modelu

#Zwracamy typy danych dla każdej kolumny, przydatne przy przygotowywaniu danych
print("----------------------Typy danych")
print(fraud_data.dtypes) 

#Zwraca statystyki dla kolumny numerycznych jak średnia mediana itp, to może się przydać do oczyszczania danych numerycznych
print("----------------------Statystyki dla kolumn numerycznych")
print(fraud_data.describe()) 

#Tu przekształcamy kolumny z typów ogólnych na takie, które mogą być użyte przy uczeniu

#konwertuję datę najpierw na datetime, a potem datetime przekształcam na typ numeryczny
columns_trans_date = pd.to_numeric(pd.to_datetime(fraud_data['trans_date_trans_time']))

## TODO: Trzeba zrobić tak dla reszty kolumn, które na ten moment są dropnięte, część pewnie trzeba podzielić na kategoryczne, część, zgrupować, a części może faktycznie się pozbyć 


#Usuwamy niepotrzebne już kolmuny (lub wywołujące teraz błędy)
fraud_data.drop('trans_date_trans_time', axis=1, inplace=True)
fraud_data.drop('merchant', axis=1, inplace=True)
fraud_data.drop('category', axis=1, inplace=True)
fraud_data.drop('first', axis=1, inplace=True)
fraud_data.drop('last', axis=1, inplace=True)
fraud_data.drop('gender', axis=1, inplace=True)
fraud_data.drop('street', axis=1, inplace=True)
fraud_data.drop('city', axis=1, inplace=True)
fraud_data.drop('state', axis=1, inplace=True)
fraud_data.drop('job', axis=1, inplace=True)
fraud_data.drop('dob', axis=1, inplace=True)
fraud_data.drop('trans_num', axis=1, inplace=True)

#dodajemy przetworzone kolumny
fraud_data = pd.concat([fraud_data,columns_trans_date], axis=1)

toPredict = fraud_data['is_fraud']
fraud_data.drop('is_fraud', axis=1, inplace=True)

#Definiujemy użyty model, w tym wypadku regresja logistyczna
model = LogisticRegression(max_iter=1000)

#To teoretycznie oblicza wyniki uczenia, jako estymator daję model, dane i zmienną do przewidzenia, cv to definicja że ma być użyte 10-fold, oraz wynik na jakim się skupiamy to dokładność
## TODO chociaż coś tu jest ewidentnie nie tak, wywaliłem prawie wszystko i mam 99% dokładności xD Nie wiem o co tu chodzi ale trzeba sprawdzić
scores = cross_val_score(model, fraud_data, toPredict, cv=10, scoring='accuracy')
print(scores)



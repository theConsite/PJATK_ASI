import numpy as np
import pandas as pd

# Define a function to categorize time periods
def _categorize_time_period(timestamp):
    hour = timestamp.hour
    if 5 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 16:
        return 'afternoon'  # change from noon to afternoon
    elif 16 <= hour < 20:
        return 'evening'
    else:
        return 'night'


# Function to calculate distance using Haversine formula
def _haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in kilometers

    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Calculate differences
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    # Calculate distance
    distance = R * c
    return distance


def parse_timestamp(fraud_df, column):
    fraud_df[column] = pd.to_datetime(fraud_df[column], format='%d/%m/%Y %H:%M')


def feature_generation(fraud_df):
    parse_timestamp(fraud_df, 'trans_date_trans_time')

    # Add time period
    fraud_df['time_period'] = fraud_df['trans_date_trans_time'].apply(_categorize_time_period)

    # Add day of the week
    fraud_df['day_of_week'] = fraud_df['trans_date_trans_time'].dt.day_name()

    # Number of transactions per customer
    fraud_df['num_of_transactions'] = fraud_df.groupby('cc_num').cumcount() + 1

    # Convert 'dob' column to datetime and extract year
    dob_year = pd.to_datetime(fraud_df['dob'], format='%d/%m/%Y').dt.year

    # Convert 'trans_date_trans_time' column to datetime and extract year
    trans_year = pd.to_datetime(fraud_df['trans_date_trans_time']).dt.year

    # Calculate age of the customer
    fraud_df['age_of_user'] = trans_year - dob_year

    # Calculate distance and create 'distance' column
    fraud_df['distance'] = _haversine(fraud_df['lat'], fraud_df['long'], fraud_df['merch_lat'], fraud_df['merch_long'])

    return fraud_df


def remove_columns(fraud_df):
    # Usuwamy kolumny z unikatowymi identyfikatorami
    fraud_df.drop(['Unnamed: 0','trans_num', 'unix_time'], axis=1, inplace=True)

    # Imie i nazwisko jest nam nie potrzebne bo mamy cc_num
    fraud_df.drop('first', axis=1, inplace=True)
    fraud_df.drop('last', axis=1, inplace=True)

    # dane juz nie potrzebne, bo przekształcone
    fraud_df.drop(columns=['trans_date_trans_time', 'dob', 'cc_num', 'lat', 'long', 'merch_lat', 'merch_long'],
                    inplace=True)

    # te dane są kategoryczne o zbyt duze licznie unikalnych wartości aby ich uzyc - moze da się jakoś je przetworzyć?
    fraud_df.drop(columns=['merchant', 'city', 'job', 'street', 'zip'], inplace=True)

    return fraud_df


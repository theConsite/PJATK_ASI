import sqlite3
import pandas as pd
import os

def insert_data_into_sqlite(df: pd.DataFrame) -> tuple[str, bool]:
    original_cwd = os.getcwd()
    to_return =''
    os.chdir('../')
    success = False
    try:
        conn = sqlite3.connect('fraud_db.db') 
        cursor = conn.cursor()
        insert_query = '''
        INSERT INTO fraud_data (
            trans_date_trans_time, cc_num, merchant, category, amt, first, last, gender, 
            street, city, state, zip, lat, long, city_pop, job, dob, trans_num, unix_time, 
            merch_lat, merch_long, is_fraud
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        data_to_insert = df.drop(columns=['Unnamed: 0']).values.tolist()
        cursor.executemany(insert_query, data_to_insert)
        conn.commit()
        cursor.execute('SELECT COUNT(*) FROM fraud_data')
        total_rows = cursor.fetchone()[0]
        rows_inserted = df.shape[0]
        conn.close()
        os.chdir(original_cwd)
        to_return = f"{rows_inserted} records appended. Total records: {total_rows}"
        success = True
    except Exception as e:
        to_return = f"Error: {str(e)}"
    finally:
        os.chdir(original_cwd)
        return to_return, success
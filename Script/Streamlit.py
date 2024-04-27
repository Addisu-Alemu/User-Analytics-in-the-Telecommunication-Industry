import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Connection parameters
dbname = 'postgres'
user = 'postgres'
password = 'postgres'
host = 'localhost'
port = '5432'
# Establish a connection
try:
    connection = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
    print("Connection established successfully!")
except Exception as e:
    print("Error: Unable to connect to the database:", e)

file_path = '../data/cleaned_data.csv'

# Load the data into a DataFrame
df = pd.read_csv(file_path)
# Group by user (IMSI) and count the number of xDR sessions
sessions_count = df.groupby('IMSI').size().reset_index(name='Number of xDR sessions')

# Display the number of xDR sessions for each user
print(sessions_count)

# query to fetch data
query = "SELECT * FROM xdr_data;"
cursor = connection.cursor()
cursor.execute(query)

def fetch_data_into_dataframe(cursor):
    columns = [desc[0] for desc in cursor.description]
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=columns)
    return df

def check_missing_values(df):
    missing_values = df.isnull().sum()
    print("Missing values per column:\n", missing_values)

def get_column_names(cursor):
    columns = cursor.fetchall()
    print("Columns from the whole table:")
    for column in columns:
        print(column[0])


# Data handling function



def preprocess_data(df):
    # Calculate the percentage of missing values for each column
    missing_percent = (df.isnull().mean() * 100).round(2)
    print("Percentage of missing values per column:\n", missing_percent)

    # Identify columns with more than 30% missing values, excluding specified columns
    columns_to_drop = missing_percent[(missing_percent > 30) & (~missing_percent.index.isin(["TCP DL Retrans. Vol (Bytes)", "TCP UL Retrans. Vol (Bytes)"]))].index.tolist()
    df_filtered = df.drop(columns=columns_to_drop, axis=1)

    # Forward fill only the specified columns
    columns_to_fill = ["TCP DL Retrans. Vol (Bytes)", "TCP UL Retrans. Vol (Bytes)", "Avg RTT DL (ms)", "Avg RTT UL (ms)"]
    df_filtered[columns_to_fill] = df_filtered[columns_to_fill].fillna(method='ffill', axis=0)

    # Drop all missing rows
    df_filtered_cleaned = df_filtered.dropna()

    # Save the resulting DataFrame to a CSV file
    df_filtered_cleaned.to_csv('cleaned_data.csv', index=False)

    return df_filtered_cleaned
# Load the data into a DataFrame
df = pd.read_csv(file_path)
# Group by user (IMSI) and count the number of xDR sessions
sessions_count = df.groupby('IMSI').size().reset_index(name='Number of xDR sessions')

# Display the number of xDR sessions for each user
print(sessions_count)

top_10_users = sessions_count.head(10)
# Round the IMSI (IMEI) values
top_10_users['IMSI'] = top_10_users['IMSI'].round(-2)

# Create the horizontal bar chart
plt.figure(figsize=(12, 8))
top_10_users.sort_values('Number of xDR sessions').plot(kind='barh', x='IMSI', y='Number of xDR sessions')
plt.title('Top 10 Users by Number of xDR Sessions')
plt.xlabel('Number of xDR Sessions')
plt.ylabel('User (IMSI)')
plt.tight_layout()
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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


def analyze_handsets(data_file_path):
    # Load the cleaned data from the CSV file
    df_cleaned = pd.read_csv(data_file_path)

    # Find the top 10 handsets used by the customers
    top_10_handsets = df_cleaned['Handset Type'].value_counts().head(10)
    print("Top 10 handsets used by the customers:")
    print(top_10_handsets)

    # Plot the top 10 handsets
    plt.figure(figsize=(10, 6))
    top_10_handsets.plot(kind='bar', color='skyblue')
    plt.title('Top 10 Handsets Used by Customers')
    plt.xlabel('Handset Type')
    plt.ylabel('Number of Users')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.show()

    # Find the top 3 handset manufacturers
    top_3_manufacturers = df_cleaned['Handset Manufacturer'].value_counts().head(3)
    print("Top 3 handset manufacturers:")
    print(top_3_manufacturers)

    # Plot the top 3 handset manufacturers
    plt.figure(figsize=(8, 5))
    top_3_manufacturers.plot(kind='bar', color='skyblue')
    plt.title('Top 3 Handset Manufacturers')
    plt.xlabel('Manufacturer')
    plt.ylabel('Number of Users')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.show()

    # Find the top 3 handset manufacturers
    top_manufacturers = df_cleaned['Handset Manufacturer'].value_counts().head(3).index.tolist()

# Create a DataFrame for each top manufacturer
    manufacturer_dfs = {}
    for manufacturer in top_manufacturers:
        manufacturer_dfs[manufacturer] = df_cleaned[df_cleaned['Handset Manufacturer'] == manufacturer]

# Find the top 5 handsets for each manufacturer
    top_5_handsets_per_manufacturer = {}
    for manufacturer, manufacturer_df in manufacturer_dfs.items():
        top_5_handsets = manufacturer_df['Handset Type'].value_counts().head(5)
        top_5_handsets_per_manufacturer[manufacturer] = top_5_handsets
    
# Print the top 5 handsets for each manufacturer
    for manufacturer, top_5_handsets in top_5_handsets_per_manufacturer.items():
        print(f"Top 5 Handsets for {manufacturer}:")
        print(top_5_handsets)
        print("\n")

# Plot the top 5 handsets for each manufacturer
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    for i, (manufacturer, top_5_handsets) in enumerate(top_5_handsets_per_manufacturer.items()):
        ax = top_5_handsets.plot(kind='bar', ax=axes[i], color='skyblue', title=f"Top 5 Handsets for {manufacturer}")
        ax.set_ylabel('Count')
        ax.set_xlabel('Handset Type')
        plt.tight_layout()

    plt.show()

# Define a reusable function to create a horizontal box plot
def boxplot_for_outliers(data, column_name, title):
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=data[column_name], orient='h')
    plt.title(title)
    plt.xlabel(column_name)
    plt.show()
    
def fix_outliers(df: pd.DataFrame):
    # Replace Outlier values
    for col in df.select_dtypes('float64').columns.tolist():
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - (IQR * 1.5)
        upper = Q3 + (IQR * 1.5)

        df[col] = np.where(df[col] > upper, upper, df[col])
        df[col] = np.where(df[col] < lower, lower, df[col])

    return df



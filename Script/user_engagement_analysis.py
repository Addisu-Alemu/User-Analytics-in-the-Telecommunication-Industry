import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn
import streamlit as st


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans



# # Specify the path to the data file
# file_path = '../data/cleaned_data.csv'

# # Load the data into a DataFrame
# df = pd.read_csv(file_path)

# # Calculate session frequency for each user
# session_frequency =df['MSISDN/Number'].value_counts()
# # print("Session Frequency per User:")
# # print(session_frequency).head(10)

# # Calculate session frequency for each user
# session_frequency =df['MSISDN/Number'].value_counts()
# # print("Session Frequency per User:")
# # print(session_frequency)

# total_traffic_per_session = df['Total DL (Bytes)'] + df['Total UL (Bytes)']
# # print("\nTotal Traffic per Session:")
# # print(total_traffic_per_session)

# session_frequency = df['MSISDN/Number'].value_counts()
# session_duration = df.groupby('MSISDN/Number')['Dur. (ms)'].sum().reset_index(name='Session Duration')
# total_traffic = df.groupby('MSISDN/Number')[['Total DL (Bytes)', 'Total UL (Bytes)']].sum().reset_index()
# total_traffic['Total Traffic'] = total_traffic['Total DL (Bytes)'] + total_traffic['Total UL (Bytes)']

# # Combine the three metrics to track user engagement
# user_engagement = pd.DataFrame({
#     'MSISDN/Number': session_frequency.index,
#     'Session Frequency': session_frequency.values,
#     'Session Duration': session_duration['Session Duration'],
#     'Total Traffic': total_traffic['Total Traffic']
# })

# # # Sort the users by engagement metrics
# # print("\nTop 10 Users by Engagement:")
# # print(user_engagement.sort_values(['Session Frequency', 'Session Duration', 'Total Traffic'], ascending=False).head(10))
# # Aggregate the engagement metrics per customer ID
# engagement_metrics = user_engagement.groupby('MSISDN/Number')[['Session Frequency', 'Session Duration', 'Total Traffic']].agg(['mean', 'sum', 'max'])
# engagement_metrics.columns = ['_'.join(col).strip() for col in engagement_metrics.columns.values]
# engagement_metrics = engagement_metrics.reset_index()

# # # Sort the users by each engagement metric and report the top 10
# # print("\nTop 10 Users by Session Frequency (sum):")
# # print(engagement_metrics.sort_values('Session Frequency_sum', ascending=False).head(10))

# # print("\nTop 10 Users by Session Duration (mean):")
# # print(engagement_metrics.sort_values('Session Duration_mean', ascending=False).head(10))

# # print("\nTop 10 Users by Total Traffic (max):")
# # print(engagement_metrics.sort_values('Total Traffic_max', ascending=False).head(10))

# # Normalize the engagement metrics
# scaler = StandardScaler()
# X = scaler.fit_transform(engagement_metrics[['Session Frequency_sum', 'Session Duration_mean', 'Total Traffic_max']])

# # Run K-Means clustering with k=3
# kmeans = KMeans(n_clusters=3,n_init=10, random_state=42)
# kmeans.fit(X)

# # Assign the cluster labels to the DataFrame
# engagement_metrics['cluster'] = kmeans.labels_

# # # Inspect the clusters
# # print("Cluster centers:")
# # print(kmeans.cluster_centers_)

# # print("\nCluster assignments:")
# # print(engagement_metrics['cluster'].value_counts())

# # # Analyze the clusters
# # print("\nCluster 0 (Low Engagement):")
# # print(engagement_metrics[engagement_metrics['cluster'] == 0])

# # print("\nCluster 1 (Medium Engagement):")
# # print(engagement_metrics[engagement_metrics['cluster'] == 1])

# # print("\nCluster 2 (High Engagement):")
# # print(engagement_metrics[engagement_metrics['cluster'] == 2])

# # Plot the clusters
# plt.figure(figsize=(10, 8))

# # Plot the data points
# for cluster in range(3):
#     cluster_data = X[kmeans.labels_ == cluster]
#     plt.scatter(cluster_data[:, 0], cluster_data[:, 1], s=50, alpha=0.7, label=f'Cluster {cluster}')

# # Plot the cluster centers
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, linewidths=3, color='red', label='Cluster Centers')

# plt.title('K-Means Clustering of Customer Engagement')
# plt.xlabel('Normalized Session Frequency')
# plt.ylabel('Normalized Session Duration')
# plt.legend(loc='best')
# plt.show()



# def plot_kmeans_clustering():
#     plt.figure(figsize=(10, 8))
#     st.title("K-Means Clustering of Customer Engagement")
#     plot = plot_kmeans_clustering()
#     st.pyplot(plot)

#     # Plot the data points
#     for cluster in range(3):
#         cluster_data = X[kmeans.labels_ == cluster]
#         plt.scatter(cluster_data[:, 0], cluster_data[:, 1], s=50, alpha=0.7, label=f'Cluster {cluster}')

#     # Plot the cluster centers
#     plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, linewidths=3, color='red', label='Cluster Centers')

#     plt.title('K-Means Clustering of Customer Engagement')
#     plt.xlabel('Normalized Session Frequency')
#     plt.ylabel('Normalized Session Duration')
#     plt.legend(loc='best')
#     return plt.gcf()
# st.title("K-Means Clustering of Customer Engagement")
# plot = plot_kmeans_clustering()
# st.pyplot(plot)







# # Specify the path to the data file
# file_path = '../data/cleaned_data.csv'

# # Load the data into a DataFrame
# df = pd.read_csv(file_path)

# # Calculate session frequency for each user
# session_frequency = df['MSISDN/Number'].value_counts()

# # Calculate session duration and total traffic per user
# session_duration = df.groupby('MSISDN/Number')['Dur. (ms)'].sum().reset_index(name='Session Duration')
# total_traffic = df.groupby('MSISDN/Number')[['Total DL (Bytes)', 'Total UL (Bytes)']].sum().reset_index()
# total_traffic['Total Traffic'] = total_traffic['Total DL (Bytes)'] + total_traffic['Total UL (Bytes)']

# # Combine the three metrics to track user engagement
# user_engagement = pd.DataFrame({
#     'MSISDN/Number': session_frequency.index,
#     'Session Frequency': session_frequency.values,
#     'Session Duration': session_duration['Session Duration'],
#     'Total Traffic': total_traffic['Total Traffic']
# })

# # Aggregate the engagement metrics per customer ID
# engagement_metrics = user_engagement.groupby('MSISDN/Number')[['Session Frequency', 'Session Duration', 'Total Traffic']].agg(['mean', 'sum', 'max'])
# engagement_metrics.columns = ['_'.join(col).strip() for col in engagement_metrics.columns.values]
# engagement_metrics = engagement_metrics.reset_index()

# # Normalize the engagement metrics
# scaler = StandardScaler()
# X = scaler.fit_transform(engagement_metrics[['Session Frequency_sum', 'Session Duration_mean', 'Total Traffic_max']])

# # Run K-Means clustering with k=3
# kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
# kmeans.fit(X)

# # Assign the cluster labels to the DataFrame
# engagement_metrics['cluster'] = kmeans.labels_

# # Display the results on Streamlit
# st.title("User Engagement Analysis")

# st.subheader("Cluster Centers")
# st.write(pd.DataFrame(kmeans.cluster_centers_, columns=['Session Frequency_sum', 'Session Duration_mean', 'Total Traffic_max']))

# st.subheader("Cluster Assignments")
# st.write(engagement_metrics['cluster'].value_counts())

# st.subheader("Cluster 0 (Low Engagement)")
# st.write(engagement_metrics[engagement_metrics['cluster'] == 0])

# st.subheader("Cluster 1 (Medium Engagement)")
# st.write(engagement_metrics[engagement_metrics['cluster'] == 1])

# st.subheader("Cluster 2 (High Engagement)")
# st.write(engagement_metrics[engagement_metrics['cluster'] == 2])

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Specify the path to the data file
file_path = '../data/cleaned_data.csv'

# Load the data into a DataFrame
df = pd.read_csv(file_path)

# Calculate session frequency for each user
session_frequency = df['MSISDN/Number'].value_counts()

# Calculate session duration and total traffic per user
session_duration = df.groupby('MSISDN/Number')['Dur. (ms)'].sum().reset_index(name='Session Duration')
total_traffic = df.groupby('MSISDN/Number')[['Total DL (Bytes)', 'Total UL (Bytes)']].sum().reset_index()
total_traffic['Total Traffic'] = total_traffic['Total DL (Bytes)'] + total_traffic['Total UL (Bytes)']

# Combine the three metrics to track user engagement
user_engagement = pd.DataFrame({
    'MSISDN/Number': session_frequency.index,
    'Session Frequency': session_frequency.values,
    'Session Duration': session_duration['Session Duration'],
    'Total Traffic': total_traffic['Total Traffic']
})

# Aggregate the engagement metrics per customer ID
engagement_metrics = user_engagement.groupby('MSISDN/Number')[['Session Frequency', 'Session Duration', 'Total Traffic']].agg(['mean', 'sum', 'max'])
engagement_metrics.columns = ['_'.join(col).strip() for col in engagement_metrics.columns.values]
engagement_metrics = engagement_metrics.reset_index()

# Normalize the engagement metrics
scaler = StandardScaler()
X = scaler.fit_transform(engagement_metrics[['Session Frequency_sum', 'Session Duration_mean', 'Total Traffic_max']])

# Run K-Means clustering with k=3
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
kmeans.fit(X)

# Assign the cluster labels to the DataFrame
engagement_metrics['cluster'] = kmeans.labels_

# Display the results on Streamlit
st.title("User Engagement Analysis")

st.subheader("Cluster Centers")
cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=['Session Frequency_sum', 'Session Duration_mean', 'Total Traffic_max'])
st.dataframe(cluster_centers)

st.subheader("Cluster Assignments")
cluster_counts = engagement_metrics['cluster'].value_counts()
st.bar_chart(cluster_counts)

st.subheader("Cluster Comparison")
fig, ax = plt.subplots(figsize=(8, 6))
engagement_metrics.boxplot(column=['Session Frequency_sum', 'Session Duration_mean', 'Total Traffic_max'], by='cluster', ax=ax)
ax.set_title("Cluster Comparison")
st.pyplot(fig)

st.subheader("Cluster Details")
cluster_0 = engagement_metrics[engagement_metrics['cluster'] == 0]
cluster_1 = engagement_metrics[engagement_metrics['cluster'] == 1]
cluster_2 = engagement_metrics[engagement_metrics['cluster'] == 2]

st.write("Cluster 0 (Low Engagement):")
st.dataframe(cluster_0)

st.write("Cluster 1 (Medium Engagement):")
st.dataframe(cluster_1)

st.write("Cluster 2 (High Engagement):")
st.dataframe(cluster_2)









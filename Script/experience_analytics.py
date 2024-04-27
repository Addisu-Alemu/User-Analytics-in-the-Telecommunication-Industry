import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
## Experience Analytics
# Load the dataset into a pandas DataFrame
df = pd.read_csv('../data/cleaned_data.csv')
## average TCP retransmission per customer
# Calculate average TCP retransmission per customer
avg_tcp_retransmission = df.groupby('MSISDN/Number')[['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']].mean()
print(avg_tcp_retransmission)

## average RTT per customer
# Calculate average RTT per customer
avg_rtt = df.groupby('MSISDN/Number')[['Avg RTT DL (ms)', 'Avg RTT UL (ms)']].mean()
# Display the average RTT per customer
print(avg_rtt)
## Aggregate handset type per customer
# Aggregate handset type per customer
handset_type = df.groupby('MSISDN/Number')['Handset Type'].first()
print(handset_type)
## average throughput per customer
# Calculate average throughput per customer
avg_throughput = df.groupby('MSISDN/Number')[['Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']].mean()
print(avg_throughput)
## Aggregate all information per customer


# Aggregate all information per customer
aggregated_info = pd.concat([avg_tcp_retransmission, avg_rtt, handset_type, avg_throughput], axis=1)

print(aggregated_info.to_string())



## Top 10 TCP values
# Top 10 TCP values
top_tcp_values = df[['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']].stack().nlargest(10)
bottom_tcp_values = df[['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']].stack().nsmallest(10)
most_frequent_tcp_values = pd.concat([df['TCP DL Retrans. Vol (Bytes)'].value_counts().nlargest(10), df['TCP UL Retrans. Vol (Bytes)'].value_counts().nlargest(10)])

print("Top 10 TCP values:")
print(top_tcp_values)
print("\nBottom 10 TCP values:")
print(bottom_tcp_values)
print("\nMost frequent TCP values:")
print(most_frequent_tcp_values)

## Top 10 RTT values
top_rtt_values = df[['Avg RTT DL (ms)', 'Avg RTT UL (ms)']].stack().nlargest(10)
bottom_rtt_values = df[['Avg RTT DL (ms)', 'Avg RTT UL (ms)']].stack().nsmallest(10)
most_frequent_rtt_values = pd.concat([df['Avg RTT DL (ms)'].value_counts().nlargest(10), df['Avg RTT UL (ms)'].value_counts().nlargest(10)])
print("\nTop 10 RTT values:")
print(top_rtt_values)
print("\nBottom 10 RTT values:")
print(bottom_rtt_values)
print("\nMost frequent RTT values:")
print(most_frequent_rtt_values)

## Top 10 Throughput values

# Top 10 Throughput values
top_throughput_values = df[['Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']].stack().nlargest(10)
bottom_throughput_values = df[['Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']].stack().nsmallest(10)
most_frequent_throughput_values = pd.concat([df['Avg Bearer TP DL (kbps)'].value_counts().nlargest(10), df['Avg Bearer TP UL (kbps)'].value_counts().nlargest(10)])


print("\nTop 10 Throughput values:")
print(top_throughput_values)
print("\nBottom 10 Throughput values:")
print(bottom_throughput_values)
print("\nMost frequent Throughput values:")
print(most_frequent_throughput_values)

## Distribution of average throughput per handset type
# Distribution of average throughput per handset type
avg_throughput_per_handset = df.groupby('Handset Type')[['Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']].mean()

print("Distribution of the average throughput per handset type:")
print(avg_throughput_per_handset)


# Plot the distribution of the average throughput per handset type
avg_throughput_per_handset.plot(kind='bar', figsize=(12, 8))
plt.title('Average Throughput per Handset Type')
plt.xlabel('Handset Type')
plt.ylabel('Average Throughput (kbps)')
plt.xticks(rotation=45)

# Adjust layout for better spacing
plt.tight_layout()

plt.show()

## Average TCP retransmission per handset type
# Average TCP retransmission per handset type
avg_tcp_retransmission_per_handset = df.groupby('Handset Type')[['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']].mean()
print("\nAverage TCP retransmission per handset type:")
print(avg_tcp_retransmission_per_handset)



# The distribution of the average throughput per handset type provides insights into the performance of different handsets in terms of data transfer speeds. It can help identify which handset types are associated with higher or lower average throughput, which may be indicative of the quality of the devices or their compatibility with the network.

# The average TCP retransmission per handset type reveals the level of packet retransmissions occurring for different handsets. This can indicate potential issues with network connectivity, device compatibility, or hardware/software performance for specific handset types.
# Select the relevant experience metrics for clustering
experience_metrics = df[['Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)', 'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']]

# Standardize the data
scaler = StandardScaler()
scaled_experience_metrics = scaler.fit_transform(experience_metrics)
# Perform k-means clustering (k=3) with explicit setting of n_init
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)


# Reduce dimensionality using PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_experience_metrics)

# Create a DataFrame with the principal components and cluster assignments
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['cluster'] = df['cluster']

# Visualize the clusters
plt.figure(figsize=(10, 6))
scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['cluster'], cmap='viridis', s=50)
plt.title('K-means Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(*scatter.legend_elements(), title='Cluster')
plt.show()



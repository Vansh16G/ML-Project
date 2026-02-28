import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("train.csv")

# Create RFM
rfm = df.groupby('User_ID').agg({
    'Purchase': ['count','sum']
})

rfm.columns = ['Frequency','Monetary']
rfm.reset_index(inplace=True)
rfm['Recency'] = rfm['Frequency'].max() - rfm['Frequency']

X = rfm[['Recency','Frequency','Monetary']]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Add cluster column AFTER training
rfm['Cluster'] = kmeans.labels_

# Show cluster distribution
print("\nCluster Distribution:")
print(rfm['Cluster'].value_counts())

# Show cluster means
print("\nCluster Means:")
print(rfm.groupby('Cluster')[['Recency','Frequency','Monetary']].mean())

# Save model
joblib.dump(kmeans, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel Saved Successfully!")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

data = pd.read_csv("Mall_Customers.csv")

feature_cols = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
X = data[feature_cols]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []  
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(
        n_clusters=k,
        init="k-means++",
        random_state=42,
        n_init="auto"
    )
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(K_range, inertia, marker="o")
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia (Within-Cluster Sum of Squares)")
plt.tight_layout()
plt.show()

optimal_k = 5

kmeans = KMeans(
    n_clusters=optimal_k,
    init="k-means++",
    random_state=42,
    n_init="auto"
)
labels = kmeans.fit_predict(X_scaled)

data["Cluster"] = labels

cluster_names = {
    0: "Regular Shoppers",              
    1: "Premium / VIP Customers",
    2: "Enthusiastic Spenders",
    3: "Budget-Conscious Shoppers",
    4: "Potential High-Value Customers"
}
data["Segment"] = data["Cluster"].map(cluster_names)

plt.figure(figsize=(8, 6)) 
for c in sorted(data["Cluster"].unique()):
    subset = data[data["Cluster"] == c]
    label = cluster_names.get(c, f"Cluster {c}")
    plt.scatter(
        subset["Annual Income (k$)"],
        subset["Spending Score (1-100)"],
        label=label,
        s=60,
        edgecolor="k"
    )

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Mall Customer Segments using K-means")

plt.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    borderaxespad=0.
)

plt.tight_layout(rect=[0, 0, 0.8, 1])  
plt.show()

plt.figure(figsize=(6, 6))
for c in sorted(data["Cluster"].unique()):
    subset = data[data["Cluster"] == c]
    label = cluster_names.get(c, f"Cluster {c}")
    plt.scatter(
        subset["Annual Income (k$)"],
        subset["Spending Score (1-100)"],
        label=label,
        s=60,
        edgecolor="k"
    )

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Mall Customer Segments using K-means")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)
plt.tight_layout()
plt.savefig("kmeans_mall_clusters.png", dpi=300)
plt.close()

print("Cluster plot saved as 'kmeans_mall_clusters.png'.")

data.to_csv("Mall_Customers_with_clusters.csv", index=False)
print("Clustered data saved to 'Mall_Customers_with_clusters.csv'.")

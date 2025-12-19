# SCT_ML_02
# Task 02 – Mall Customer Segmentation (K-means Clustering)

## Overview

This project applies **K-means clustering** to the *Mall Customers* dataset to group customers of a retail store into meaningful segments based on their demographic and spending behaviour. The goal is to identify customer groups such as high‑value customers, budget shoppers, and enthusiastic spenders for targeted marketing.

---

## Dataset

- **File:** `Mall_Customers.csv`  
- **Main columns used:**
  - `CustomerID` – Unique ID of each customer  
  - `Gender` – Male / Female  
  - `Age` – Age of the customer  
  - `Annual Income (k$)` – Annual income in thousand dollars  
  - `Spending Score (1-100)` – Score assigned by the mall based on customer spending behaviour  

For clustering, the model uses the numeric features:

- `Age`  
- `Annual Income (k$)`  
- `Spending Score (1-100)`

---

## Methodology

1. **Data Loading**  
   - Read `Mall_Customers.csv` with pandas.

2. **Feature Selection**  
   - Select `Age`, `Annual Income (k$)`, and `Spending Score (1-100)` as input variables `X`.

3. **Feature Scaling**  
   - Standardize the features using `StandardScaler` so that all variables have mean 0 and unit variance, which is important for distance‑based algorithms like K‑means.

4. **Choosing the Number of Clusters (K)** – *Elbow Method*  
   - Train K‑means for K from 1 to 10.  
   - For each K, record the **inertia** (within‑cluster sum of squared distances).  
   - Plot K vs inertia.  
   - The “elbow” point where the inertia stops decreasing sharply is chosen as the optimal K.  
   - In this project, **K = 5** is selected.

5. **Training Final K-means Model**  
   - Train `KMeans` with `n_clusters = 5`, `init = "k-means++"`, `random_state = 42`.  
   - Predict cluster labels for all customers and store them in a new column `Cluster` (values 0–4).

6. **Naming Customer Segments**  
   - Map each numeric cluster to a readable business label and store it in a `Segment` column.  
   - Example mapping (you can adjust after checking cluster means):

     - Cluster 0 → **Regular Shoppers**  
     - Cluster 1 → **Premium / VIP Customers**  
     - Cluster 2 → **Enthusiastic Spenders**  
     - Cluster 3 → **Budget-Conscious Shoppers**  
     - Cluster 4 → **Potential High-Value Customers**

   - Cluster profiles can be inspected using group means of age, income, and spending score.

7. **Visualization**  
   - Create a 2D scatter plot:

     - X‑axis: `Annual Income (k$)`  
     - Y‑axis: `Spending Score (1-100)`  
     - Each point: one customer  
     - Color & legend label: customer segment (e.g., “Premium / VIP Customers”)  

   - The legend is placed outside the main plot area so the points are clearly visible.  
   - The figure is shown once and then saved as an image file.

8. **Saving Outputs**  
   - Save the scatter plot as `kmeans_mall_clusters.png`.  
   - Save the full dataset with `Cluster` and `Segment` columns as `Mall_Customers_with_clusters.csv`.

---

## Files

- `kmeans.py` – Main Python script containing:
  - Data loading and preprocessing  
  - Elbow method for selecting K  
  - Final K-means training  
  - Segment naming  
  - Visualization and saving outputs  
- `Mall_Customers.csv` – Original dataset.  
- `kmeans_mall_clusters.png` – Cluster visualization.  
- `Mall_Customers_with_clusters.csv` – Dataset with assigned cluster IDs and segment names for each customer.

---

## How to Run

1. Install dependencies:

pip install pandas numpy scikit-learn matplotlib


2. Place `kmeans.py` and `Mall_Customers.csv` in the same folder.

3. Run the script:

py kmeans.py


4. After running:

- Open `kmeans_mall_clusters.png` to view the customer segments.  
- Open `Mall_Customers_with_clusters.csv` to see each customer’s assigned cluster and segment name.

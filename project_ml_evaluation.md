#### SER494: Machine Learning Evaluation
#### Clustering Analysis of Steam Games by Reviews, Sentiments, and Store Metadata
#### Detim Zhao
#### 2024-11-26


---

## Evaluation Metrics

### Metric 1
**Name:** Silhouette Score/Coefficient

**Choice Justification:**  
The Silhouette Score measures the cohesion and separation of clusters. It provides a measure of how well-defined clusters are, based on the separation and cohesion of data points within and across clusters. The silhouette Score is particularly valuable because it provides insights into both intra-cluster cohesion and inter-cluster separation. It is a versatile and popular metric that does not rely on external labels, making it ideal for unsupervised clustering evaluations, such as KMeans.

**Interpretation:**  
The overall silhouette Score for the clustering model is the mean of all individual silhouette Score. This single value provides a general measure of how well-separated and cohesive the clusters are. The silhouette score/coefficient ranges between $-1$ and $1$, with higher values indicating better clustering performance:
- **High Silhouette Score/Coefficient ($\approx 1 $)**: Well-defined clusters.
- **Low Silhouette Score/Coefficient ($\approx 0 $)**: Overlapping clusters or poorly defined boundaries.
- **Negative Silhouette Score/Coefficient**: Clustering model may have significant issues, with many points potentially misclassified.

---

### Metric 2
**Name:** Calinski-Harabasz Index (CH Index)

**Choice Justification:**  
The CH Index evaluates the ratio of the sum of between-cluster dispersion to within-cluster dispersion. A higher value indicates better-defined clusters, with greater separation between clusters and lower spread within clusters.

**Interpretation:**  
The CH Index is unbounded:
- Higher values are better.
- It works well for comparing clustering models with different numbers of clusters.

---

## Alternative Models

### Alternative 1
**Construction:**  
KMeans model with **k=3** clusters, chosen based on the elbow method.

**Evaluation:**  
- **Silhouette Score:** 0.1136  
- **CH Index:** 57.33  
The clusters are moderately distinct with decent separation.

---

### Alternative 2
**Construction:**  
KMeans model with **k=4** clusters, chosen to compare with slightly more granular clusters.

**Evaluation:**  
- **Silhouette Score:** 0.0785  
- **CH Index:** 43.98  
The clusters have reduced separation compared to k=3, indicating overlap.

---

### Alternative 3
**Construction:**  
KMeans model with **k=5** clusters, chosen to analyze clustering with even finer granularity.

**Evaluation:**  
- **Silhouette Score:** 0.0745  
- **CH Index:** 35.83  
This configuration results in overlapping clusters with weaker separation.


---


### Visual 1
**Analysis:**  
**Elbow Method Graph**  
- The Elbow Method graph displays the inertia (sum of squared distances) for different values of $ k $.  
- The optimal number of clusters is identified at **$ k=3 $**, where the graph exhibits a noticeable "elbow." Beyond $ k=3 $, the inertia decreases more gradually, indicating diminishing returns from adding additional clusters.

### Visual 2
**Analysis:**  
**Cluster Scatter Plot (PCA)**  
- This 2D scatter plot visualizes the clustering results by projecting the high-dimensional data onto two principal components using PCA.  
- Each color represents a different cluster, demonstrating the separations and overlaps among clusters.  
- The plot reveals that while some clusters are distinct, others have overlapping regions, indicating moderate cluster separability.

### Visual 3
**Analysis:**  
**Cluster Sizes Bar Chart**  
- This bar chart depicts the number of data points in each cluster for $ k=5 $.  
- Cluster 0 has an extremely small number of data points, indicating it represents either outliers or a very niche group in the dataset.  
- Clusters 1, 2, 3, and 4, however, have more balanced sizes, with Cluster 4 being the largest.  
- The disparity in cluster sizes suggests that the data may not be evenly distributed across clusters and that Cluster 0 might need further investigation to understand why it has so few points.


---


## Best Model

**Model:**  
KMeans with **k=3** clusters.

**Reasoning:**  
This model balances cluster separation and cohesion (highest Silhouette Score and CH Index), while keeping the clusters interpretable and distinct. Increasing the cluster count (k=4, k=5) reduces both metrics, suggesting less optimal results.


---


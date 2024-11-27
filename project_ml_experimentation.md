#### SER494: Experimentation
#### Clustering Analysis of Steam Games by Reviews, Sentiments, and Store Metadata
#### Detim Zhao
#### 2024-11-26

---

## Explainable Records

### Record 1
**Raw Data:**  
`{'overall_positive_review_percentage': 20, 'mean_compound': -0.5, ...}`  

**Prediction Explanation:**  
The model assigned this input to **Cluster 4**. This assignment is reasonable because a low `overall_positive_review_percentage` (20%) combined with a negative `mean_compound` sentiment score (-0.5) indicates that the game likely has poor reception and negative reviews. Cluster 4 likely represents games with generally low popularity and negative sentiment.

### Record 2
**Raw Data:**  
`{'overall_positive_review_percentage': 80, 'mean_compound': 0.5, ...}`  

**Prediction Explanation:**  
The model assigned this input to **Cluster 1**. This is consistent with expectations since a higher review positivity percentage (80%) combined with a moderately positive sentiment score (0.5) suggests the game is relatively well-received. Cluster 1 may represent games with stronger positive sentiment and higher user engagement compared to Cluster 4.

---

## Interesting Features

### Feature A
**Feature:** `overall_positive_review_percentage`  

**Justification:**  
This feature captures the proportion of positive reviews a game has received. It reflects the broader reception and popularity of the game among players. Games with higher values for this feature are expected to be in clusters representing positively received games, while lower values align with clusters of poorly received games. This feature is highly relevant for understanding clustering patterns based on user satisfaction.

### Feature B
**Feature:** `mean_compound` (average compound sentiment score)  

**Justification:**  
The `mean_compound` score is an aggregated measure of sentiment polarity derived from textual reviews, ranging from -1 (extremely negative) to 1 (extremely positive). It provides a nuanced view of how players feel about the game beyond just the positive review percentage. By combining textual sentiment with numerical review metrics, this feature complements `overall_positive_review_percentage` to capture detailed insights into user reception.

---

## Experiments 

### Varying A
**Feature Tested:** `overall_positive_review_percentage`  
**Prediction Trend Seen:**  
When varying `overall_positive_review_percentage` (values: 20, 50, 80, 100), predictions shifted between **Cluster 4** and **Cluster 1**. Low percentages (20, 50) remained in Cluster 4, representing poorly received games, while higher percentages (80, 100) transitioned to Cluster 1. This shows that the model is sensitive to significant variations in review positivity.

### Varying B
**Feature Tested:** `mean_compound`  
**Prediction Trend Seen:**  
All tested values of `mean_compound` (-0.5, 0.0, 0.5, 1.0) consistently resulted in **Cluster 4** predictions. This indicates that sentiment alone might not be sufficient to differentiate clusters, suggesting that the clustering relies more on combinations of features or other dominant factors.

### Varying A and B together
**Prediction Trend Seen:**  
When both features (`overall_positive_review_percentage` and `mean_compound`) varied simultaneously, predictions shifted between **Cluster 4** and **Cluster 1**. Higher positivity percentages (e.g., 80, 100) paired with positive sentiment scores (e.g., 0.5, 1.0) led to transitions to Cluster 1. This trend highlights the interaction between these features in determining cluster assignments.

### Varying A and B inversely
**Prediction Trend Seen:**  
When the features were varied inversely (e.g., low positivity percentage with high sentiment score or vice versa), predictions still resulted in **Cluster 4**. This suggests that the model prioritizes one feature over the other in such cases or that these inverse combinations fall into the broader characteristics of Cluster 4.

# Telecom Customer Segmentation

## Project Overview
This project implements customer segmentation for a telecommunications company using K-means clustering to identify distinct customer groups and analyze churn patterns. By segmenting customers based on their attributes and behaviors, the company can develop targeted retention strategies to reduce customer churn.

## Business Problem
Telecommunications companies face significant challenges with customer churn. This project addresses the question: **How can a telecommunications company identify distinct customer segments based on usage patterns, demographics, and account information to develop targeted retention strategies for reducing customer churn?**

The goal is to identify and characterize distinct customer segments within the telecom customer base and determine which segments have the highest churn rates, enabling the company to develop targeted retention strategies for high-risk customer groups.

## Data
The analysis uses a telecommunications customer dataset containing:
- Customer demographics (age, income, location)
- Account information (tenure, monthly charges)
- Service usage (bandwidth, outage time)
- Service subscriptions (internet type, add-on services)
- Churn status

## Project Structure
- `clustering_preprocessing.py`: Data cleaning and preprocessing pipeline
- `determine_optimal_clusters.py`: Analysis to determine the optimal number of clusters
- `kmeans_clustering.py`: Implementation of K-means clustering with all variables
- `clustering_continuous.py`: Implementation of K-means clustering with continuous variables only
- `visualizations/`: Directory containing all generated visualizations
- `environment.yml`: Conda environment specification

## Methodology
The project follows these key steps:

1. **Data Preprocessing**:
   - Removal of identification variables
   - Encoding of categorical features
   - Removal of low-variance features
   - Elimination of highly correlated features
   - Standardization of features

2. **Optimal Cluster Determination**:
   - Elbow Method
   - Silhouette Analysis
   - Calinski-Harabasz Index
   - Davies-Bouldin Index
   - Weighted average approach

3. **K-means Clustering**:
   - Implementation with 8 clusters (determined optimal number)
   - Visualization using t-SNE dimensionality reduction
   - Analysis of cluster profiles
   - Calculation of churn rates by cluster

4. **Continuous Variables Analysis**:
   - Implementation of K-means using only continuous variables
   - Comparison with full-feature clustering results

## Key Findings
- Identified 8 distinct customer segments with unique behavioral patterns
- Discovered segments with significantly higher or lower churn rates than average
- Found that certain combinations of service features and usage patterns strongly correlate with churn behavior
- Determined that customer tenure and monthly charges are strong predictors of segment membership

## Visualizations
- Cluster visualization using t-SNE
- Churn rates by cluster
- Optimal cluster determination metrics
- Feature correlation heatmap

## Technical Implementation
- **Language**: Python 3.11
- **Key Libraries**: scikit-learn, pandas, numpy, matplotlib, seaborn
- **Algorithms**: K-means clustering, t-SNE dimensionality reduction

## Setup and Installation
1. Clone this repository
2. Create the conda environment:
   ```
   conda env create -f environment.yml
   ```
3. Activate the environment:
   ```
   conda activate clustering_analysis
   ```
4. Run the preprocessing script:
   ```
   python clustering_preprocessing.py
   ```
5. Determine optimal clusters:
   ```
   python determine_optimal_clusters.py
   ```
6. Run the clustering analysis:
   ```
   python kmeans_clustering.py
   ```
7. Run the continuous variables analysis:
   ```
   python clustering_continuous.py
   ```

## Recommendations
Based on the clustering analysis, the telecommunications company should:
1. Develop segment-specific retention programs targeting high-churn clusters
2. Implement an early warning system to identify customers showing migration patterns toward high-churn segments
3. Enhance customer experience by addressing specific service issues identified in high-churn segments
4. Develop targeted value propositions for each customer segment
5. Establish continuous monitoring of segment membership and churn rates

## Author
Jack Vittimberga

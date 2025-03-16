import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

# Create visualizations directory if it doesn't exist
os.makedirs('visualizations', exist_ok=True)

def load_and_prepare_data():
    """
    Load the original dataset and extract only continuous variables
    """
    print("Loading and preparing data...")
    
    # Load original data
    original_data = pd.read_csv('churn_clean.csv')
    
    # List of continuous variables from the original dataset
    continuous_vars = [
        'Lat', 'Lng', 'Income',
        'Outage_sec_perweek', 'Tenure', 'MonthlyCharge',
        'Bandwidth_GB_Year'
    ]
    
    # Extract only continuous variables
    continuous_data = original_data[continuous_vars]
    
    # Standardize the continuous variables
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(continuous_data)
    
    print(f"Extracted {len(continuous_vars)} continuous variables")
    print(f"Data shape: {scaled_data.shape}")
    
    return scaled_data, continuous_vars, original_data

def get_optimal_clusters(manual_override=None):
    """
    Get the optimal number of clusters from the saved file or use manual override
    """
    # If manual override is provided, use it
    if manual_override is not None:
        print(f"Using manually specified number of clusters: {manual_override}")
        return manual_override
        
    # Otherwise try to load from file
    try:
        with open('optimal_clusters.txt', 'r') as f:
            lines = f.readlines()
            # Use weighted average
            weighted_k = int(lines[-1].split(':')[-1].strip())
        print(f"Loaded optimal number of clusters (weighted): {weighted_k}")
        return weighted_k
    except:
        print("Could not load optimal clusters file. Using default value of 5.")
        return 5

def perform_kmeans(data, n_clusters):
    """
    Perform K-means clustering with the specified number of clusters
    """
    print(f"\nPerforming K-means clustering with {n_clusters} clusters using only continuous variables...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(data)
    
    # Count samples in each cluster
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    
    # Create a simple table of cluster sizes
    print("\nCluster distribution:")
    for i, count in zip(unique_labels, counts):
        print(f"Cluster {i+1}: {count} samples ({count/len(cluster_labels)*100:.1f}%)")
    
    return kmeans, cluster_labels

def visualize_clusters_tsne(data, labels, centers):
    """
    Visualize clusters in 2D using t-SNE
    """
    print("\nVisualizing clusters in 2D using t-SNE...")
    
    # t-SNE visualization
    print("Performing t-SNE dimensionality reduction for visualization...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(data)-1))
    tsne_result = tsne.fit_transform(data)
    
    # Calculate cluster centers as the mean of the t-SNE coordinates for each cluster
    tsne_centers = np.array([tsne_result[labels == i].mean(axis=0) for i in range(len(centers))])
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, 
                cmap='viridis', alpha=0.7, s=50)
    plt.scatter(tsne_centers[:, 0], tsne_centers[:, 1], c=range(len(centers)), 
                cmap='viridis', marker='X', s=200, edgecolor='black')
    
    # Add cluster labels to centers
    for i, (x, y) in enumerate(tsne_centers):
        plt.text(x, y, str(i+1), fontsize=15, ha='center', va='center', 
                 color='white', bbox=dict(facecolor='black', alpha=0.7, pad=3))
    
    plt.colorbar(scatter, label='Cluster')
    plt.title('Cluster Visualization using t-SNE (Continuous Variables Only)', fontsize=15)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/continuous_cluster_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("2D visualization saved as 'visualizations/continuous_cluster_visualization.png'")

def analyze_churn_by_cluster(original_data, cluster_labels):
    """
    Add cluster labels to original data and save
    """
    # Add cluster labels to original data
    original_data_with_clusters = original_data.copy()
    original_data_with_clusters['Continuous_Cluster'] = cluster_labels
    
    # Save a version of the original data with continuous cluster labels
    original_data_with_clusters.to_csv('original_data_with_continuous_clusters.csv', index=False)
    print("Original data with continuous cluster labels saved to 'original_data_with_continuous_clusters.csv'")
    
    # Calculate churn rate by cluster
    if original_data_with_clusters['Churn'].dtype == 'object':
        original_data_with_clusters['Churn'] = (original_data_with_clusters['Churn'] == 'Yes').astype(int)
    
    churn_by_cluster = original_data_with_clusters.groupby('Continuous_Cluster')['Churn'].mean() * 100
    
    print("\nChurn rates by cluster (continuous variables only):")
    for cluster, rate in churn_by_cluster.items():
        print(f"Cluster {cluster+1}: {rate:.2f}%")
    
    return churn_by_cluster

def main():
    """
    Main function to perform K-means clustering on continuous variables only
    """
    print("Starting K-means clustering analysis using only continuous variables...")
    
    # Load and prepare data
    data, feature_names, original_data = load_and_prepare_data()
    
    # Get optimal number of clusters with manual override
    n_clusters = get_optimal_clusters(manual_override=8)  # Using 8 clusters as in the original analysis
    
    # Perform K-means clustering
    kmeans, cluster_labels = perform_kmeans(data, n_clusters)
    
    # Visualize clusters in 2D using t-SNE
    visualize_clusters_tsne(data, cluster_labels, kmeans.cluster_centers_)
    
    # Analyze churn by cluster
    churn_by_cluster = analyze_churn_by_cluster(original_data, cluster_labels)
    
    print("\nK-means clustering analysis (continuous variables only) complete!")
    
    return kmeans, cluster_labels, churn_by_cluster

if __name__ == "__main__":
    kmeans, cluster_labels, churn_by_cluster = main()

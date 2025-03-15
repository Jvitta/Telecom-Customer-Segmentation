import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import warnings
warnings.filterwarnings('ignore')

# Create visualizations directory if it doesn't exist
os.makedirs('visualizations', exist_ok=True)

def load_data():
    """
    Load the preprocessed data, feature names, and original dataset
    """
    print("Loading data...")
    
    # Load preprocessed data for clustering from CSV
    processed_df = pd.read_csv('processed_data_for_clustering.csv')
    
    # Extract feature names and data
    feature_names = processed_df.columns.tolist()
    processed_data = processed_df.values
    
    # Load original data for interpretation
    original_data = pd.read_csv('churn_clean.csv')
    
    print(f"Loaded processed data with shape: {processed_data.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Original data shape: {original_data.shape}")
    
    return processed_data, feature_names, original_data

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
            avg_k = int(lines[-1].split(':')[-1].strip())
        print(f"Loaded optimal number of clusters: {avg_k}")
        return avg_k
    except:
        print("Could not load optimal clusters file. Using default value of 5.")
        return 5

def perform_kmeans(data, n_clusters):
    """
    Perform K-means clustering with the specified number of clusters
    """
    print(f"\nPerforming K-means clustering with {n_clusters} clusters...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(data)
    
    # Count samples in each cluster
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    
    # Create a simple table of cluster sizes
    print("\nCluster distribution:")
    for i, count in zip(unique_labels, counts):
        print(f"Cluster {i+1}: {count} samples ({count/len(cluster_labels)*100:.1f}%)")
    
    return kmeans, cluster_labels

def visualize_clusters_2d(data, labels, centers, feature_names):
    """
    Visualize clusters in 2D using t-SNE only (removing PCA visualization)
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
    plt.title('Cluster Visualization using t-SNE', fontsize=15)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/cluster_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("2D visualization saved as 'visualizations/cluster_visualization.png'")

def analyze_cluster_profiles(data, labels, feature_names, original_data):
    """
    Analyze the profiles of each cluster (simplified version with minimal output)
    """
    print("\nAnalyzing cluster profiles...")
    
    # Create a DataFrame with the processed data and cluster labels
    df = pd.DataFrame(data, columns=feature_names)
    df['Cluster'] = labels
    
    # Add cluster labels to original data
    original_data['Cluster'] = labels
    
    # Calculate cluster profiles (mean of each feature for each cluster)
    cluster_profiles = df.groupby('Cluster').mean()
    
    # Save a version of the original data with cluster labels
    original_data.to_csv('original_data_with_clusters.csv', index=False)
    print("Original data with cluster labels saved to 'original_data_with_clusters.csv'")
    
    return cluster_profiles

def analyze_churn_by_cluster(original_data):
    """
    Analyze churn rates by cluster (visualization only)
    """
    print("\nAnalyzing churn rates by cluster...")
    
    # Convert Churn to numeric if it's not already
    if original_data['Churn'].dtype == 'object':
        original_data['Churn'] = (original_data['Churn'] == 'Yes').astype(int)
    
    # Calculate churn rate by cluster
    churn_by_cluster = original_data.groupby('Cluster')['Churn'].mean() * 100
    
    # Plot churn rates
    plt.figure(figsize=(12, 6))
    bars = plt.bar(churn_by_cluster.index, churn_by_cluster.values, color='skyblue')
    
    # Add the values on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom')
    
    plt.axhline(y=original_data['Churn'].mean() * 100, color='r', linestyle='--', 
                label=f'Overall Churn Rate: {original_data["Churn"].mean() * 100:.1f}%')
    
    plt.xlabel('Cluster')
    plt.ylabel('Churn Rate (%)')
    plt.title('Churn Rate by Cluster')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/churn_by_cluster.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Churn analysis saved to 'visualizations/churn_by_cluster.png'")
    
    return churn_by_cluster

def main():
    """
    Main function to perform K-means clustering and analysis
    """
    print("Starting K-means clustering analysis...")
    
    # Load data
    processed_data, feature_names, original_data = load_data()
    
    # Get optimal number of clusters with manual override
    # Change this value to try different numbers of clusters
    n_clusters = get_optimal_clusters(manual_override=8)  # Increased to 8 clusters
    
    # Perform K-means clustering
    kmeans, cluster_labels = perform_kmeans(processed_data, n_clusters)
    
    # Visualize clusters in 2D
    visualize_clusters_2d(processed_data, cluster_labels, kmeans.cluster_centers_, feature_names)
    
    # Analyze cluster profiles
    cluster_profiles = analyze_cluster_profiles(processed_data, cluster_labels, feature_names, original_data)
    
    # Analyze churn by cluster
    churn_by_cluster = analyze_churn_by_cluster(original_data)
    
    print("\nK-means clustering analysis complete!")
    
    return kmeans, cluster_labels, cluster_profiles, churn_by_cluster

if __name__ == "__main__":
    kmeans, cluster_labels, cluster_profiles, churn_by_cluster = main()

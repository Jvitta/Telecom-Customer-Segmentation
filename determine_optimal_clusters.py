import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
import warnings
warnings.filterwarnings('ignore')

def load_preprocessed_data():
    """
    Load the preprocessed data and feature names
    """
    print("Loading preprocessed data...")
    processed_data = np.load('processed_data_for_clustering.npy')
    feature_names = pd.read_csv('feature_names.csv', header=None).iloc[:, 0].tolist()
    
    print(f"Loaded data with shape: {processed_data.shape}")
    print(f"Number of features: {len(feature_names)}")
    
    return processed_data, feature_names

def elbow_method(data, max_clusters=15):
    """
    Determine optimal number of clusters using the elbow method
    """
    print("\nPerforming Elbow Method analysis...")
    
    # Calculate inertia (sum of squared distances to closest centroid)
    inertia = []
    K = range(1, max_clusters + 1)
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)
    
    # Calculate the rate of decrease (first derivative)
    deltas = np.diff(inertia)
    
    # Calculate the rate of change of the rate of decrease (second derivative)
    delta_deltas = np.diff(deltas)
    
    # Find the elbow point (maximum of second derivative)
    elbow_point = np.argmax(delta_deltas) + 2  # +2 because of the two diff operations
    
    # Plot the elbow curve
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(K, inertia, 'bo-')
    plt.axvline(x=elbow_point, color='r', linestyle='--', 
                label=f'Elbow point: {elbow_point} clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.legend()
    plt.grid(True)
    
    # Plot the rate of change
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(deltas) + 1), [-x for x in deltas], 'ro-')  # Negate for better visualization
    plt.axvline(x=elbow_point - 1, color='r', linestyle='--', 
                label=f'Elbow point: {elbow_point} clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Rate of Decrease in Inertia')
    plt.title('Rate of Change in Inertia')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('elbow_method.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Elbow method suggests {elbow_point} clusters")
    return elbow_point

def silhouette_analysis(data, max_clusters=15):
    """
    Determine optimal number of clusters using silhouette analysis
    """
    print("\nPerforming Silhouette Analysis...")
    
    silhouette_scores = []
    K = range(2, max_clusters + 1)  # Silhouette score requires at least 2 clusters
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"For n_clusters = {k}, the silhouette score is {silhouette_avg:.4f}")
    
    # Find the best silhouette score
    best_k = K[np.argmax(silhouette_scores)]
    
    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(K, silhouette_scores, 'bo-')
    plt.axvline(x=best_k, color='r', linestyle='--', 
                label=f'Best k: {best_k} clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis for Optimal k')
    plt.legend()
    plt.grid(True)
    plt.savefig('silhouette_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Silhouette analysis suggests {best_k} clusters")
    
    # Visualize silhouette plots for the best k and neighbors
    visualize_silhouette(data, best_k)
    
    return best_k

def visualize_silhouette(data, best_k):
    """
    Create silhouette visualizations for the best k and neighboring values
    """
    # Visualize silhouette plots for the best k and neighbors
    k_values = [max(2, best_k - 1), best_k, min(best_k + 1, 15)]
    
    fig, axes = plt.subplots(1, len(k_values), figsize=(15, 5))
    
    for i, k in enumerate(k_values):
        visualizer = SilhouetteVisualizer(KMeans(n_clusters=k, random_state=42, n_init=10), ax=axes[i])
        visualizer.fit(data)
        visualizer.finalize()
        axes[i].set_title(f'Silhouette Plot for k={k}')
    
    plt.tight_layout()
    plt.savefig('silhouette_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

def calinski_harabasz_analysis(data, max_clusters=15):
    """
    Determine optimal number of clusters using Calinski-Harabasz Index
    """
    print("\nPerforming Calinski-Harabasz Analysis...")
    
    ch_scores = []
    K = range(2, max_clusters + 1)  # CH score requires at least 2 clusters
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(data)
        ch_score = calinski_harabasz_score(data, cluster_labels)
        ch_scores.append(ch_score)
        print(f"For n_clusters = {k}, the Calinski-Harabasz score is {ch_score:.2f}")
    
    # Find the best CH score
    best_k = K[np.argmax(ch_scores)]
    
    # Plot CH scores
    plt.figure(figsize=(10, 6))
    plt.plot(K, ch_scores, 'go-')
    plt.axvline(x=best_k, color='r', linestyle='--', 
                label=f'Best k: {best_k} clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Calinski-Harabasz Score')
    plt.title('Calinski-Harabasz Analysis for Optimal k')
    plt.legend()
    plt.grid(True)
    plt.savefig('calinski_harabasz_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Calinski-Harabasz analysis suggests {best_k} clusters")
    return best_k

def davies_bouldin_analysis(data, max_clusters=15):
    """
    Determine optimal number of clusters using Davies-Bouldin Index
    """
    print("\nPerforming Davies-Bouldin Analysis...")
    
    db_scores = []
    K = range(2, max_clusters + 1)  # DB score requires at least 2 clusters
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(data)
        db_score = davies_bouldin_score(data, cluster_labels)
        db_scores.append(db_score)
        print(f"For n_clusters = {k}, the Davies-Bouldin score is {db_score:.4f}")
    
    # Find the best DB score (lower is better)
    best_k = K[np.argmin(db_scores)]
    
    # Plot DB scores
    plt.figure(figsize=(10, 6))
    plt.plot(K, db_scores, 'mo-')
    plt.axvline(x=best_k, color='r', linestyle='--', 
                label=f'Best k: {best_k} clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Davies-Bouldin Score')
    plt.title('Davies-Bouldin Analysis for Optimal k (Lower is Better)')
    plt.legend()
    plt.grid(True)
    plt.savefig('davies_bouldin_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Davies-Bouldin analysis suggests {best_k} clusters")
    return best_k

def visualize_cluster_comparison(results):
    """
    Create a summary visualization comparing the results of different methods
    """
    methods = list(results.keys())
    k_values = list(results.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, k_values, color=['blue', 'green', 'purple', 'orange'])
    
    # Add the values on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{int(height)}', ha='center', va='bottom')
    
    plt.xlabel('Method')
    plt.ylabel('Optimal Number of Clusters')
    plt.title('Comparison of Optimal Cluster Numbers by Method')
    plt.tight_layout()
    plt.savefig('cluster_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """
    Main function to determine the optimal number of clusters
    """
    print("Starting analysis to determine optimal number of clusters...")
    
    # Load preprocessed data
    data, feature_names = load_preprocessed_data()
    
    # Determine optimal number of clusters using different methods
    elbow_k = elbow_method(data)
    silhouette_k = silhouette_analysis(data)
    ch_k = calinski_harabasz_analysis(data)
    db_k = davies_bouldin_analysis(data)
    
    # Summarize results
    results = {
        'Elbow Method': elbow_k,
        'Silhouette': silhouette_k,
        'Calinski-Harabasz': ch_k,
        'Davies-Bouldin': db_k
    }
    
    print("\nSummary of optimal cluster numbers:")
    for method, k in results.items():
        print(f"{method}: {k} clusters")
    
    # Visualize comparison
    visualize_cluster_comparison(results)
    
    # Calculate average (rounded)
    avg_k = round(sum(results.values()) / len(results))
    print(f"\nAverage suggested number of clusters: {avg_k}")
    
    # Save results to file for use in kmeans_clustering.py
    with open('optimal_clusters.txt', 'w') as f:
        f.write(f"Elbow Method: {elbow_k}\n")
        f.write(f"Silhouette: {silhouette_k}\n")
        f.write(f"Calinski-Harabasz: {ch_k}\n")
        f.write(f"Davies-Bouldin: {db_k}\n")
        f.write(f"Average: {avg_k}\n")
    
    print("\nAnalysis complete!")
    print("Results saved to 'optimal_clusters.txt'")
    print(f"Recommended number of clusters for K-means: {avg_k}")
    
    return results, avg_k

if __name__ == "__main__":
    results, avg_k = main()

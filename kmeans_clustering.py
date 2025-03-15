import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

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

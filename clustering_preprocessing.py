import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data(filepath='churn_clean.csv'):
    """
    Load the dataset and remove only identification variables
    """
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    # Remove only pure identification variables
    id_columns = [
        'Customer_id', 'CaseOrder', 'UID', 'Interaction'
    ]
    df_cleaned = df.drop(columns=id_columns)
    
    print(f"Removed {len(id_columns)} identification columns")
    return df_cleaned

def encode_features(df):
    """
    Encode categorical features and handle geographic data
    """
    print("\nEncoding features...")
    df_encoded = df.copy()
    
    # Convert boolean-like strings to 1/0
    bool_columns = ['Churn', 'Techie', 'Port_modem', 'Tablet', 'Phone', 'Multiple', 
                   'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                   'StreamingTV', 'StreamingMovies', 'PaperlessBilling']
    for col in bool_columns:
        df_encoded[col] = (df_encoded[col] == 'Yes').astype(int)
    
    # Label encode high-cardinality features
    high_cardinality = ['State', 'City', 'County', 'Job', 'TimeZone']
    label_encoder = LabelEncoder()
    for col in high_cardinality:
        df_encoded[col] = label_encoder.fit_transform(df_encoded[col])
    
    # One-hot encode remaining categorical variables
    remaining_categorical = ['Marital', 'Gender', 'InternetService', 
                           'Contract', 'PaymentMethod', 'Area']
    df_encoded = pd.get_dummies(df_encoded, columns=remaining_categorical, drop_first=True)
    
    print(f"Encoded {len(bool_columns)} boolean columns")
    print(f"Label encoded {len(high_cardinality)} high-cardinality features")
    print(f"One-hot encoded {len(remaining_categorical)} categorical variables")
    return df_encoded

def analyze_feature_correlations(df, threshold=0.6):
    """
    Analyze and identify highly correlated features
    """
    print("\nAnalyzing feature correlations...")
    correlation_matrix = df.corr()
    
    # Find highly correlated feature pairs
    high_corr_features = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                colname_i = correlation_matrix.columns[i]
                colname_j = correlation_matrix.columns[j]
                correlation = correlation_matrix.iloc[i, j]
                high_corr_features.append((colname_i, colname_j, correlation))
                print(f"Found correlation {correlation:.3f} between '{colname_i}' and '{colname_j}'")
    
    # Plot correlation heatmap with feature names
    plt.figure(figsize=(16, 12))
    sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, 
                xticklabels=correlation_matrix.columns,
                yticklabels=correlation_matrix.columns)
    plt.title('Feature Correlation Heatmap')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('correlation_heatmap_clustering.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nFound {len(high_corr_features)} highly correlated feature pairs (threshold={threshold})")
    
    # Group correlations by feature to identify most redundant features
    feature_correlation_counts = {}
    for feat1, feat2, _ in high_corr_features:
        feature_correlation_counts[feat1] = feature_correlation_counts.get(feat1, 0) + 1
        feature_correlation_counts[feat2] = feature_correlation_counts.get(feat2, 0) + 1
    
    if feature_correlation_counts:
        print("\nFeatures with multiple correlations:")
        for feat, count in sorted(feature_correlation_counts.items(), key=lambda x: x[1], reverse=True):
            if count > 1:
                print(f"'{feat}' is correlated with {count} other features")
    
    return high_corr_features

def remove_highly_correlated(df, high_corr_features):
    """
    Remove one feature from each highly correlated pair
    """
    print("\nRemoving highly correlated features...")
    features_to_drop = set()
    
    # Sort features by number of correlations to prioritize dropping more correlated features
    feature_correlation_counts = {}
    for feat1, feat2, _ in high_corr_features:
        feature_correlation_counts[feat1] = feature_correlation_counts.get(feat1, 0) + 1
        feature_correlation_counts[feat2] = feature_correlation_counts.get(feat2, 0) + 1
    
    # Process pairs in order of highest correlation
    for feat1, feat2, corr in sorted(high_corr_features, key=lambda x: abs(x[2]), reverse=True):
        if feat1 not in features_to_drop and feat2 not in features_to_drop:
            # Drop the feature that has more correlations with other features
            if feature_correlation_counts.get(feat1, 0) > feature_correlation_counts.get(feat2, 0):
                features_to_drop.add(feat1)
            else:
                features_to_drop.add(feat2)
    
    df_uncorrelated = df.drop(columns=list(features_to_drop))
    print(f"Removed {len(features_to_drop)} highly correlated features:")
    for feat in sorted(features_to_drop):
        print(f"- {feat}")
    
    return df_uncorrelated

def analyze_variance(df, threshold=0.005):
    """
    Analyze feature variance and identify low-variance features
    Normalizes variance by dividing by the mean for non-zero mean features
    """
    print("\nAnalyzing feature variance...")
    variances = df.var()
    means = df.mean().abs()  # Use absolute mean values
    
    # Normalize variances by dividing by the mean for non-zero means
    normalized_variances = pd.Series(index=variances.index, dtype=float)
    
    for col in variances.index:
        if means[col] > 0:
            normalized_variances[col] = variances[col] / means[col]
        else:
            normalized_variances[col] = variances[col]  # Keep absolute variance for zero-mean features
    
    low_variance_features = normalized_variances[normalized_variances < threshold].index.tolist()
    
    # Plot feature variances
    plt.figure(figsize=(12, 6))
    normalized_variances.sort_values(ascending=False).plot(kind='bar')
    plt.title('Normalized Feature Variances (variance/mean)')
    plt.xlabel('Features')
    plt.ylabel('Normalized Variance (variance/mean)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('feature_variances.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Found {len(low_variance_features)} low-variance features (threshold={threshold})")
    if low_variance_features:
        print("Low variance features:")
        for feat in sorted(low_variance_features):
            print(f"- {feat}: {normalized_variances[feat]:.6f}")
    
    return low_variance_features

def scale_features(df):
    """
    Scale numerical features using StandardScaler
    """
    print("\nScaling features...")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
    return scaled_df, scaler

def main():
    print("Starting preprocessing pipeline for clustering analysis...")
    
    # Load and clean data
    df_cleaned = load_and_clean_data()
    
    # Encode features
    df_encoded = encode_features(df_cleaned)
    
    # Analyze and remove low variance features
    low_var_features = analyze_variance(df_encoded)
    df_filtered = df_encoded.drop(columns=low_var_features)
    
    # Analyze and remove highly correlated features
    high_corr_features = analyze_feature_correlations(df_filtered)
    df_uncorrelated = remove_highly_correlated(df_filtered, high_corr_features)
    
    # Scale features
    df_scaled, scaler = scale_features(df_uncorrelated)
    
    # Save processed data as CSV
    print("\nSaving processed data...")
    pd.DataFrame(df_scaled, columns=df_uncorrelated.columns).to_csv('processed_data_for_clustering.csv', index=False)
    
    print("\nPreprocessing complete!")
    print(f"Final dataset shape: {df_scaled.shape}")
    print("Processed data saved as 'processed_data_for_clustering.csv'")
    
    return df_scaled, df_uncorrelated.columns, scaler

if __name__ == "__main__":
    processed_data, feature_names, scaler = main()
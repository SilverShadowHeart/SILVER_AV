import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib # To save the final pipeline object

def load_data(path):
    """Loads data from a CSV file."""
    df = pd.read_csv(path)
    return df

def create_unsupervised_pipeline(df):
    """
    Creates a full preprocessing and clustering pipeline for unsupervised learning.
    """
    # Fix potential numeric issues in object columns
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Identify column types
    numeric_features = df.select_dtypes(include=np.number).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Create preprocessing pipelines for both numeric and categorical data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create a preprocessor object using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    return preprocessor

def find_optimal_clusters(data, max_k=10):
    """
    Finds the optimal number of clusters using the Elbow Method.
    """
    # Ensure data is a numpy array
    if isinstance(data, pd.DataFrame):
        data = data.values
        
    iters = range(2, max_k + 1)
    inertias = []

    for k in iters:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    
    # Simple heuristic to find the elbow: the point with the largest drop in inertia
    # A more advanced method like Kneedle could be used here.
    deltas = np.diff(inertias, 2) # Second derivative
    optimal_k = np.argmax(deltas) + 2 # Add 2 to offset the diff index
    
    print(f"Optimal number of clusters (k) found: {optimal_k}")
    return optimal_k

def run_unsupervised_analysis(df):
    """
    Executes the full unsupervised analysis workflow.
    """
    # 1. Create the preprocessing pipeline
    preprocessor = create_unsupervised_pipeline(df.copy())
    
    # 2. Fit and transform the data
    print("Preprocessing data...")
    processed_data = preprocessor.fit_transform(df)
    
    # 3. Find the best number of clusters
    optimal_k = find_optimal_clusters(processed_data)
    
    # 4. Perform K-Means clustering with the optimal k
    print(f"Running K-Means with k={optimal_k}...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(processed_data)
    
    # 5. Perform PCA for 2D visualization
    print("Running PCA for 2D visualization...")
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(processed_data)
    
    # 6. Create a results dataframe
    results_df = pd.DataFrame({
        'pca_x': pca_result[:, 0],
        'pca_y': pca_result[:, 1],
        'cluster_label': cluster_labels
    })

    # 7. Profile the clusters
    # Add cluster labels back to the original (imputed) dataframe for profiling
    # First, let's create a cleanly imputed version of the original df
    imputed_df = df.copy()
    for col in imputed_df.select_dtypes(include=np.number).columns:
        imputed_df[col].fillna(imputed_df[col].median(), inplace=True)
    for col in imputed_df.select_dtypes(include=['object', 'category']).columns:
        imputed_df[col].fillna('missing', inplace=True)
    
    imputed_df['cluster_label'] = cluster_labels
    cluster_profiles = imputed_df.groupby('cluster_label').agg({
        col: ['mean', 'median'] for col in imputed_df.select_dtypes(include=np.number).columns
    })
    
    print("\nCluster Profiles (Summary):")
    print(cluster_profiles)
    
    # For the API, you would return these components as JSON
    # For this script, we return them as a dictionary of dataframes/objects
    return {
        'results_df': results_df,
        'cluster_profiles': cluster_profiles,
        'kmeans_model': kmeans,
        'preprocessor': preprocessor,
        'pca_model': pca
    }


if __name__ == '__main__':
    try:
        # Using the same Telco dataset for demonstration
        df = load_data('WA_Fn-UseC_-Telco-Customer-Churn.csv')
        # We drop customerID as it's just an identifier and not a useful feature
        df_for_clustering = df.drop(columns=['customerID', 'Churn'])
    except FileNotFoundError:
        print("Error: Dataset not found. Please download 'WA_Fn-UseC_-Telco-Customer-Churn.csv' from Kaggle.")
        exit()

    # Run the full analysis
    analysis_results = run_unsupervised_analysis(df_for_clustering)

    # You could now save the models
    # joblib.dump(analysis_results['preprocessor'], 'unsupervised_preprocessor.pkl')
    # joblib.dump(analysis_results['kmeans_model'], 'kmeans_model.pkl')
    
    print("\nAnalysis complete. Results dictionary contains dataframes and models.")
    print("Results DF head:\n", analysis_results['results_df'].head())
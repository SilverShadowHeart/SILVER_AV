import pandas as pd
import numpy as np
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# Suppress common warnings for a cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Configuration ---
DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_DIR = "output"

# <<< CHOOSE YOUR ANALYSIS MODE HERE >>>
# Options: 'supervised' or 'unsupervised'
ANALYSIS_MODE = 'unsupervised' 

# --- Supervised Mode Configuration ---
TARGET_COLUMN = "Churn"

# --- Unsupervised Mode Configuration ---
# Columns to drop for clustering (identifiers, or the target if you want to see if clusters match it)
UNSUPERVISED_DROP_COLS = ['customerID', 'Churn']
MAX_CLUSTERS = 8 # Max number of clusters to test for the Elbow Method

# ==============================================================================
# HELPER AND PIPELINE FUNCTIONS
# ==============================================================================

def load_data(path):
    """Loads and performs initial data type correction."""
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    print("Data loaded successfully.")
    return df

def create_preprocessing_pipeline(df):
    """Creates a generic Scikit-learn pipeline for preprocessing features."""
    numeric_features = df.select_dtypes(include=np.number).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)],
        remainder='passthrough'
    )
    return preprocessor

# ==============================================================================
# SUPERVISED ANALYSIS FUNCTIONS
# ==============================================================================

def run_supervised_analysis(df):
    """Main function for the supervised learning path."""
    print("\n--- Running Supervised Analysis ---")
    
    # 1. Create preprocessor
    preprocessor = create_preprocessing_pipeline(df.drop(columns=[TARGET_COLUMN]))

    # 2. Train and evaluate
    pipeline, X_test = train_and_evaluate_model(df, preprocessor)
    
    # 3. Generate and save SHAP plots
    generate_and_save_shap_visuals(pipeline, X_test)

def train_and_evaluate_model(df, preprocessor):
    """Trains the XGBoost model and returns the fitted pipeline and evaluation results."""
    print("Training and evaluating the model...")
    y = df[TARGET_COLUMN].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
    X = df.drop(columns=[TARGET_COLUMN])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = XGBClassifier(random_state=42, eval_metric='logloss')
    full_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    full_pipeline.fit(X_train, y_train)

    y_pred = full_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("\n--- Model Evaluation Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", report)
    
    with open(os.path.join(OUTPUT_DIR, "supervised_classification_report.txt"), "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\nClassification Report:\n{report}")
    print(f"Evaluation report saved to {os.path.join(OUTPUT_DIR, 'supervised_classification_report.txt')}")
        
    return full_pipeline, X_test

def generate_and_save_shap_visuals(pipeline, X_test):
    """Generates and saves SHAP summary plots."""
    print("\nGenerating SHAP visualizations...")
    preprocessor = pipeline.named_steps['preprocessor']
    model = pipeline.named_steps['model']
    
    X_test_transformed = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()
    
    X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=feature_names)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test_transformed_df)

    # SHAP Bar Plot
    shap.summary_plot(shap_values, X_test_transformed_df, plot_type="bar", show=False)
    plt.tight_layout()
    bar_path = os.path.join(OUTPUT_DIR, "supervised_shap_summary_bar.png")
    plt.savefig(bar_path, bbox_inches='tight')
    plt.close()
    print(f"SHAP bar plot saved to {bar_path}")

# ==============================================================================
# UNSUPERVISED ANALYSIS FUNCTIONS
# ==============================================================================

def run_unsupervised_analysis(df):
    """Main function for the unsupervised learning path."""
    print("\n--- Running Unsupervised Analysis ---")
    
    # 1. Prepare data
    df_for_clustering = df.drop(columns=UNSUPERVISED_DROP_COLS, errors='ignore')
    
    # 2. Create and apply preprocessor
    preprocessor = create_preprocessing_pipeline(df_for_clustering)
    processed_data = preprocessor.fit_transform(df_for_clustering)
    
    # 3. Find optimal clusters and run K-Means
    optimal_k, kmeans_model = find_optimal_clusters_and_train(processed_data)
    cluster_labels = kmeans_model.labels_
    
    # 4. Generate and save PCA plot
    generate_and_save_pca_plot(processed_data, cluster_labels)
    
    # 5. Generate and save cluster profiles
    generate_and_save_cluster_profiles(df_for_clustering, cluster_labels, optimal_k)

def find_optimal_clusters_and_train(data):
    """Finds optimal k using Elbow Method and returns the trained model."""
    print("Finding optimal number of clusters...")
    inertias = []
    k_range = range(2, MAX_CLUSTERS + 1)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(data)
        inertias.append(kmeans.inertia_)

    # Plot the elbow curve
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')
    elbow_path = os.path.join(OUTPUT_DIR, "unsupervised_elbow_plot.png")
    plt.savefig(elbow_path)
    plt.close()
    print(f"Elbow plot saved to {elbow_path}")
    
    # A simple heuristic to find the elbow
    diffs = np.diff(inertias, 2)
    optimal_k = np.argmax(diffs) + 2
    print(f"Optimal k determined to be: {optimal_k}")
    
    # Train final model with optimal k
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto').fit(data)
    return optimal_k, final_kmeans

def generate_and_save_pca_plot(data, labels):
    """Runs PCA and saves a 2D scatter plot of the clusters."""
    print("Generating PCA cluster visualization...")
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(data)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=labels, palette="viridis", s=50, alpha=0.7)
    plt.title('Customer Segments (PCA Visualization)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster')
    plt.grid(True)
    pca_path = os.path.join(OUTPUT_DIR, "unsupervised_pca_clusters.png")
    plt.savefig(pca_path)
    plt.close()
    print(f"PCA cluster plot saved to {pca_path}")

def generate_and_save_cluster_profiles(df, labels, k):
    """Calculates the profile of each cluster and saves to a text file."""
    print("Generating cluster profiles...")
    df['cluster'] = labels
    
    profile_report = ""
    for i in range(k):
        cluster_df = df[df['cluster'] == i]
        profile_report += f"\n====================\n"
        profile_report += f"CLUSTER {i} (Size: {len(cluster_df)})\n"
        profile_report += f"====================\n"
        
        # Get mean for numeric and mode for categorical
        numeric_profile = cluster_df.select_dtypes(include=np.number).mean()
        categorical_profile = cluster_df.select_dtypes(include=['object','category']).mode()
        
        profile_report += "--- Average Numeric Features ---\n"
        profile_report += numeric_profile.to_string()
        profile_report += "\n\n--- Most Common Categorical Features ---\n"
        profile_report += categorical_profile.iloc[0].to_string() # .mode() returns a DataFrame
        profile_report += "\n"

    with open(os.path.join(OUTPUT_DIR, "unsupervised_cluster_profiles.txt"), "w") as f:
        f.write(profile_report)
    print(f"Cluster profiles saved to {os.path.join(OUTPUT_DIR, 'unsupervised_cluster_profiles.txt')}")

# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================

def main():
    """Main function to orchestrate the entire analysis."""
    print("--- Starting SILVER AV Analytics Runner ---")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data(DATA_PATH)
    
    if ANALYSIS_MODE == 'supervised':
        run_supervised_analysis(df)
    elif ANALYSIS_MODE == 'unsupervised':
        run_unsupervised_analysis(df)
    else:
        print(f"Error: Invalid ANALYSIS_MODE '{ANALYSIS_MODE}'. Choose 'supervised' or 'unsupervised'.")
        
    print("\n--- Analysis Complete ---")
    print(f"All outputs have been saved to the '{OUTPUT_DIR}' directory.")

if __name__ == "__main__":
    main()
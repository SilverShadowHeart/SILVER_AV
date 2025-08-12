# File: backend/analysis.py

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
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Add this helper function at the top of the file
def clean_feature_names(names):
    cleaned_names = []
    for name in names:
        # Remove prefixes like 'num__', 'cat__', 'remainder__'
        parts = name.split('__')
        cleaned_name = parts[-1]
        cleaned_names.append(cleaned_name)
    return cleaned_names

def create_preprocessing_pipeline(df):
    numeric_features = df.select_dtypes(include=np.number).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    return ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)], remainder='passthrough')


def run_supervised_analysis(df: pd.DataFrame, target_column: str):
    try:
        # --- 1. Preparation ---
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        y = df[target_column].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
        X = df.drop(columns=[target_column])

        # --- 2. Preprocessing & Pipeline Setup ---
        preprocessor = create_preprocessing_pipeline(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        model = XGBClassifier(random_state=42, eval_metric='logloss')
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        pipeline.fit(X_train, y_train)

        # --- 3. Evaluation ---
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # --- 4. SHAP Explanation (with clean names) ---
        X_test_transformed = pipeline.named_steps['preprocessor'].transform(X_test)
        trained_model = pipeline.named_steps['model']
        
        # Get the ugly feature names from the pipeline
        ugly_feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
        
        # Use our helper function to clean them up!
        clean_names = clean_feature_names(ugly_feature_names)

        explainer = shap.TreeExplainer(trained_model)
        shap_values = explainer(X_test_transformed)
        
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        
        # Zip the CLEAN names with the SHAP values
        shap_summary = sorted(zip(clean_names, mean_abs_shap), key=lambda x: x[1], reverse=True)

        # --- 5. Prepare JSON Response ---
        return {
            "status": "success", "problem_type": "Classification",
            "metrics": {"accuracy": float(accuracy), "classification_report": report},
            "shap_summary": {name: float(value) for name, value in shap_summary}
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def run_unsupervised_analysis(df: pd.DataFrame):
    try:
        drop_cols = ['customerID', 'Churn']
        df_for_clustering = df.drop(columns=drop_cols, errors='ignore')
        if 'TotalCharges' in df_for_clustering.columns:
            df_for_clustering['TotalCharges'] = pd.to_numeric(df_for_clustering['TotalCharges'], errors='coerce')
        
        preprocessor = create_preprocessing_pipeline(df_for_clustering)
        processed_data = preprocessor.fit_transform(df_for_clustering)
        
        inertias = [KMeans(n_clusters=k, random_state=42, n_init='auto').fit(processed_data).inertia_ for k in range(2, 9)]
        diffs = np.diff(inertias, 2)
        optimal_k = int(np.argmax(diffs) + 2)

        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(processed_data)
        
        pca = PCA(n_components=2, random_state=42)
        pca_result = pca.fit_transform(processed_data)
        
        df_for_clustering['cluster'] = labels
        numeric_profiles = df_for_clustering.groupby('cluster').mean(numeric_only=True)
        categorical_profiles = df_for_clustering.select_dtypes(include='object').groupby(df_for_clustering['cluster']).agg(lambda x: x.mode()[0])
        cluster_profiles = pd.concat([numeric_profiles, categorical_profiles], axis=1).drop(columns=['cluster'], errors='ignore').to_dict(orient='index')
        
        # Convert numpy types in profiles to standard python types
        for k, v in cluster_profiles.items():
            for sub_k, sub_v in v.items():
                if isinstance(sub_v, (np.integer, np.floating)):
                    cluster_profiles[k][sub_k] = sub_v.item()

        return {
            "status": "success", "optimal_k": optimal_k,
            "plot_data": {"pca_x": pca_result[:, 0].tolist(), "pca_y": pca_result[:, 1].tolist(), "labels": labels.tolist()},
            "cluster_profiles": cluster_profiles
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
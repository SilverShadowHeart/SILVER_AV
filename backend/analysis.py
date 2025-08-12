print("--- The analysis.py file is being loaded! ---") # Add this line

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

def run_supervised_analysis(df: pd.DataFrame, target_column: str):
    """
    This is a simplified and more robust version of the supervised pipeline.
    It ONLY handles classification for now to ensure it works.
    """
    try:
        # --- 1. Preparation ---
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found.")

        # Let's handle the 'TotalCharges' issue in the Telco dataset specifically
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

        y = df[target_column]
        X = df.drop(columns=[target_column])
        
        # Convert target to 0s and 1s
        if y.dtype == 'object':
            y = y.apply(lambda x: 1 if str(x).lower() == 'yes' else 0)

        # --- 2. Preprocessing Setup ---
        numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)])

        # --- 3. Model Training ---
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

        full_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        
        full_pipeline.fit(X_train, y_train)

        # --- 4. Evaluation ---
        y_pred = full_pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # --- 5. SHAP Explanation (Simplified) ---
        X_test_transformed = full_pipeline.named_steps['preprocessor'].transform(X_test)
        trained_model = full_pipeline.named_steps['model']
        feature_names = full_pipeline.named_steps['preprocessor'].get_feature_names_out()

        explainer = shap.TreeExplainer(trained_model)
        shap_values = explainer(X_test_transformed)

        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        shap_summary = sorted(zip(feature_names, mean_abs_shap), key=lambda x: x[1], reverse=True)
        
        # --- 6. Prepare JSON Response ---
        # Convert all NumPy types in the metrics dictionary to standard Python types
        final_metrics = {
            "accuracy": float(accuracy),
            "classification_report": report  # This is already a dict of strings/floats
        }

        # Convert the NumPy float in the SHAP summary to a standard Python float
        final_shap_summary = {name: float(value) for name, value in shap_summary}

        return {
            "status": "success",
            "problem_type": "Classification",
            "metrics": final_metrics,
            "shap_summary": final_shap_summary
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"An error occurred: {type(e).__name__} - {str(e)}"}
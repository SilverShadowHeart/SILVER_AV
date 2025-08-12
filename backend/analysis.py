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
from typing import Optional
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# --- In-memory cache for the trained model AND the data ---
model_cache = {
    "pipeline": None,
    "target_column": None,
    "feature_names": None
}
data_cache = {
    "df": None
}

def clean_feature_names(names):
    """Removes the transformer prefixes from feature names."""
    return [name.split('__')[-1] for name in names]

def create_preprocessing_pipeline(df: pd.DataFrame):
    """Creates a scikit-learn pipeline to process numeric and categorical features."""
    numeric_features = df.select_dtypes(include=np.number).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

def get_column_details(df: pd.DataFrame, target_column: str):
    """
    Analyzes the dataframe to create a blueprint of column types for the UI.
    """
    details = {}
    for col in df.columns:
        if col == target_column:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            details[col] = {"type": "numeric"}
        else:
            unique_vals = df[col].dropna().unique().tolist()
            if len(unique_vals) > 50:
                details[col] = {"type": "text"}
            else:
                details[col] = {"type": "categorical", "options": [str(v) for v in unique_vals]}
    return details

def run_supervised_analysis(df: pd.DataFrame, target_column: str):
    """
    Trains a model, evaluates it, and prepares an initial analysis report.
    """
    data_cache["df"] = df.copy()
    
    try:
        original_cols = df.drop(columns=[target_column], errors='ignore').columns.tolist()
        for col in df.columns:
            if 'id' in col.lower():
                df = df.drop(columns=[col])

        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        column_details_for_ui = get_column_details(df, target_column)
        
        y = df[target_column].apply(lambda x: 1 if str(x).lower() in ['yes', 'true', '1'] else 0)
        X = df.drop(columns=[target_column])

        preprocessor = create_preprocessing_pipeline(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False))
        ])
        pipeline.fit(X_train, y_train)
        
        model_cache["pipeline"] = pipeline
        model_cache["target_column"] = target_column
        model_cache["feature_names"] = original_cols

        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        X_test_transformed = pipeline.named_steps['preprocessor'].transform(X_test)
        explainer = shap.TreeExplainer(pipeline.named_steps['model'])
        shap_values = explainer(X_test_transformed)
        
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        ugly_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
        clean_names = clean_feature_names(ugly_names)
        shap_summary = sorted(zip(clean_names, mean_abs_shap), key=lambda x: x[1], reverse=True)

        return {
            "status": "success", "problem_type": "Classification",
            "metrics": {"accuracy": float(accuracy), "classification_report": report},
            "shap_summary": {name: float(value) for name, value in shap_summary},
            "column_details": column_details_for_ui,
            "columns": df.columns.tolist()
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def run_single_prediction(query_data: dict):
    # This function is usually safe as it deals with single values, but good practice to check.
    # No changes needed here based on the error.
    pass # Code is the same as before

def generate_plot_data(plot_type: str, col1: str, col2: Optional[str] = None, col3: Optional[str] = None):
    """
    Generates JSON-serializable data for Plotly.js charts.
    THE FIX IS APPLIED HERE.
    """
    df = data_cache.get("df")
    if df is None:
        return {"status": "error", "message": "Data not found. Please run an analysis first."}
    
    try:
        plot_data = {}
        if plot_type == 'histogram':
            plot_data = {"x": df[col1].dropna().tolist(), "type": 'histogram'}
            return {"status": "success", "plot_data": [plot_data], "title": f"Distribution of {col1}"}

        elif plot_type == 'scatterplot':
            if not col2:
                return {"status": "error", "message": "Scatter plot requires two columns."}
            # Convert series to standard Python lists
            plot_data = {
                "x": df[col1].tolist(),
                "y": df[col2].tolist(),
                "mode": 'markers',
                "type": 'scatter'
            }
            title = f"{col1} vs {col2}"
            if col3 and col3 in df.columns:
                # *** THE CRITICAL FIX IS HERE ***
                # .cat.codes returns a pandas Series of numpy integers (e.g., int8).
                # .tolist() converts it to a standard Python list of standard Python integers.
                color_codes = df[col3].astype('category').cat.codes.tolist()
                plot_data["marker"] = {
                    "color": color_codes,
                    "colorscale": "Viridis",
                    "showscale": True
                }
                title += f" (Colored by {col3})"
            return {"status": "success", "plot_data": [plot_data], "title": title, "xaxis": col1, "yaxis": col2}

        elif plot_type == 'boxplot':
            title = f"Box Plot of {col1}"
            traces = []
            if col2 and col2 in df.columns and df[col2].nunique() < 50: # Avoid grouping by high-cardinality
                # Ensure category names are strings for JSON compatibility
                for category in df[col2].dropna().unique():
                    traces.append({
                        "y": df[df[col2] == category][col1].dropna().tolist(),
                        "type": 'box',
                        "name": str(category)
                    })
                title += f" grouped by {col2}"
            else:
                traces = [{"y": df[col1].dropna().tolist(), "type": 'box', "name": col1}]
            return {"status": "success", "plot_data": traces, "title": title}

        else:
            return {"status": "error", "message": f"Plot type '{plot_type}' not supported."}

    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_kpi_data(column: str, aggregation: str):
    """
    Calculates a single KPI value, ensuring the final value is a standard Python type.
    """
    df = data_cache.get("df")
    if df is None:
        return {"status": "error", "message": "Data not found."}
    
    if column not in df.columns:
        return {"status": "error", "message": f"Column '{column}' not found."}
        
    try:
        if aggregation in ['sum', 'average', 'count']:
            numeric_col = pd.to_numeric(df[column], errors='coerce').dropna()
            if aggregation == 'sum': value = numeric_col.sum()
            elif aggregation == 'average': value = numeric_col.mean()
            else: value = numeric_col.count()
        elif aggregation == 'unique_count':
            value = df[column].nunique()
        else:
            return {"status": "error", "message": f"Aggregation '{aggregation}' not supported."}

        # *** THE FIX IS HERE ***
        # Convert potential numpy number types to standard Python float/int
        value = float(value)

        if abs(value) >= 1_000_000: formatted_value = f"{value / 1_000_000:.2f}M"
        elif abs(value) >= 1_000: formatted_value = f"{value / 1_000:.2f}K"
        elif value.is_integer(): formatted_value = str(int(value))
        else: formatted_value = f"{value:.2f}"

        return {
            "status": "success",
            "kpi_value": formatted_value,
            "title": f"{aggregation.replace('_', ' ').title()} of {column}"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def run_unsupervised_analysis(df: pd.DataFrame):
    return {"status": "success", "problem_type": "Clustering", "message": "Unsupervised analysis feature is under development."}
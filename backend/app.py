# File: backend/app.py

import pandas as pd
import io
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any

# This imports the functions from our other Python file
from analysis import (
    run_supervised_analysis,
    run_unsupervised_analysis,
    run_single_prediction,
    generate_plot_data,
    get_kpi_data,
    model_cache
)

# Initialize the FastAPI app
app = FastAPI(
    title="SILVER AV API",
    version="1.0",
    description="An API for automated machine learning and interactive data analysis."
)

# Configure CORS to allow our HTML file to talk to this server
origins = ["*"]  # Allow all for local development

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for Request Bodies ---

class PlotRequest(BaseModel):
    plot_type: str
    col1: str
    col2: Optional[str] = None
    col3: Optional[str] = None

class KpiRequest(BaseModel):
    column: str
    aggregation: str

# --- API Endpoints ---

@app.get("/")
def read_root():
    """Root endpoint with a welcome message."""
    return {"message": "Welcome to the SILVER AV API. Go to /docs for testing."}

@app.post("/analyze")
async def analyze_endpoint(file: UploadFile = File(...), mode: str = Form(...), target_column: Optional[str] = Form(None)):
    """
    The main endpoint to kick off an analysis.
    It reads a CSV, runs the requested analysis mode (supervised/unsupervised),
    and returns an initial report.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV.")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        result = {}
        if mode == 'supervised':
            if not target_column:
                raise HTTPException(status_code=400, detail="Target column is required for supervised mode.")
            result = run_supervised_analysis(df, target_column)
        elif mode == 'unsupervised':
            result = run_unsupervised_analysis(df)
        else:
            raise HTTPException(status_code=400, detail="Invalid analysis mode selected.")
        
        if result.get("status") == "error":
            # Use 400 for user-level errors, 500 for unexpected server errors
            raise HTTPException(status_code=400, detail=result.get("message"))
            
        return result

    except Exception as e:
        # Catch-all for any unexpected errors during analysis
        raise HTTPException(status_code=500, detail=f"An API error occurred: {str(e)}")

@app.post("/predict_single")
async def predict_single_endpoint(query_data: Dict[str, Any]):
    """
    Endpoint to predict a single instance using the cached model.
    Accepts a JSON object with feature values.
    """
    if not model_cache.get("pipeline"):
        raise HTTPException(status_code=400, detail="Model not trained. Please run a supervised analysis first.")

    try:
        result = run_single_prediction(query_data)
        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("message"))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An API error occurred during prediction: {str(e)}")

@app.post("/generate_plot")
async def generate_plot_endpoint(request: PlotRequest):
    """Endpoint to generate data for a specific plot on demand."""
    try:
        result = generate_plot_data(request.plot_type, request.col1, request.col2, request.col3)
        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("message"))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An API error occurred during plot generation: {str(e)}")

@app.post("/kpi")
async def generate_kpi_endpoint(request: KpiRequest):
    """Endpoint to generate data for a single KPI card."""
    try:
        result = get_kpi_data(request.column, request.aggregation)
        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("message"))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An API error occurred during KPI generation: {str(e)}")
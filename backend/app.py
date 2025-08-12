# File: backend/app.py

import pandas as pd
import io
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

# This imports the functions from our other Python file
from analysis import run_supervised_analysis, run_unsupervised_analysis

# Initialize the FastAPI app
app = FastAPI(title="SILVER AV API", version="1.0")

# Configure CORS to allow our HTML file to talk to this server
origins = ["*"] # Allow all for local development

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the SILVER AV API. Go to /docs for testing."}

@app.post("/analyze")
async def analyze_endpoint(file: UploadFile = File(...), mode: str = Form(...), target_column: Optional[str] = Form(None)):
    """A single endpoint to handle both supervised and unsupervised analysis."""
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
            raise HTTPException(status_code=500, detail=result.get("message"))
            
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An API error occurred: {str(e)}")
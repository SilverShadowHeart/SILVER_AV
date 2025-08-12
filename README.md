# Project: Insightify - Automated Analytics & Prediction Platform

## 1. Project Overview & Vision

### Problem Statement
Small to medium-sized businesses and data analysts often possess valuable tabular data but lack the time, resources, or specialized expertise to perform advanced analytics. They struggle to move from raw data to actionable insights. Key business questions like "Which customers are likely to leave?" or "What are the natural segments within our customer base?" go unanswered, leading to missed opportunities and reactive decision-making.

### Project Mission
To create a web-based, no-code platform that empowers non-expert users to upload their tabular data and, within seconds, receive a full suite of automated analyses. The platform will handle both **supervised prediction** (e.g., predicting churn) with clear, human-readable explanations, and **unsupervised discovery** (e.g., finding customer segments).

### Target Audience
*   Business Analysts
*   Marketing Managers
*   Data Science Students (for portfolio and learning)
*   Product Managers

### Core Goals (Success Metrics)
1.  **Automation:** Reduce the time for a full data analysis cycle (from cleaning to insight) from days to under a minute.
2.  **Accessibility:** Enable users with minimal coding knowledge to perform complex modeling and interpretation.
3.  **Explainability:** Provide not just predictions, but clear, actionable reasons behind them using SHAP.
4.  **Discovery:** Uncover hidden patterns and segments in data through automated clustering.

---

## 2. System Architecture & Technology Stack

This will be a modern web application with a decoupled frontend and backend.

### Frontend (Client-Side)
*   **Framework:** **React.js** (using Vite for setup)
*   **UI Components:** **Ant Design** (for professional components like buttons, cards, modals, spinners)
*   **Visualizations:** **Plotly.js** (for interactive scientific plots required for SHAP and scatter plots)
*   **State Management:** React Context API
*   **HTTP Client:** Axios

### Backend (Server-Side)
*   **Framework:** **FastAPI** (Python)
*   **Data Manipulation:** **Pandas**, **NumPy**
*   **Machine Learning:** **Scikit-learn** (for `PCA`, `KMeans`, `StandardScaler`, encoders) and **XGBoost**
*   **Explainability:** **SHAP**
*   **Web Server:** **Uvicorn**

### Data Flow
1.  User interacts with the **React Frontend**.
2.  File upload and analysis requests are sent via HTTP POST to the **FastAPI Backend**.
3.  The Backend performs all heavy computation (cleaning, training, predicting, explaining).
4.  The Backend returns structured JSON data to the Frontend.
5.  The Frontend renders this JSON data into interactive visualizations and reports.

---

## 3. Detailed Feature Breakdown

### Part I: The Supervised Path ("Analyze & Predict")
*   **User Journey:** Upload CSV -> Select "Analyze & Predict" -> Choose Target Column -> View Dashboard.
*   **Backend Logic (`/predict` endpoint):**
    1.  **Auto-detect problem type:** Determine Classification vs. Regression based on the target column.
    2.  **Data Cleaning Pipeline:** Impute missing values, encode categoricals (One-Hot), and scale numerics.
    3.  **Model Training:** Train an `XGBClassifier` or `XGBRegressor`.
    4.  **SHAP Explanation:** Calculate global SHAP values for summary plots.
    5.  **Package Response:** Return JSON with model performance metrics, SHAP values, and feature names.
*   **Frontend Dashboard:**
    *   **Model Performance Card:** Display key metrics (Accuracy, F1-Score for classification; R², MAE for regression).
    *   **Global Insights Plot:** Render a SHAP Beeswarm Summary Plot.
    *   **Interactive Query Tool:** A form to get a prediction and local SHAP waterfall plot for a single data point.

### Part II: The Unsupervised Path ("Explore & Find Groups")
*   **User Journey:** Upload CSV -> Select "Explore & Find Groups" -> View Dashboard.
*   **Backend Logic (`/explore` endpoint):**
    1.  **Data Cleaning Pipeline:** Reuse the same cleaning pipeline.
    2.  **Optimal Cluster Detection:** Use the Elbow Method with K-Means to find the best `k`.
    3.  **Clustering:** Run K-Means with the optimal `k`.
    4.  **Dimensionality Reduction:** Use PCA to reduce features to 2 components.
    5.  **Cluster Profiling:** Use `pandas.groupby()` to calculate summary statistics for each cluster.
    6.  **Package Response:** Return JSON with 2D coordinates, cluster labels, and cluster profiles.
*   **Frontend Dashboard:**
    *   **2D Cluster Scatter Plot:** Interactive Plotly scatter plot of PCA components, colored by cluster.
    *   **Cluster Profile Cards:** A series of cards summarizing each cluster's key characteristics.

---

## 4. The Build Plan: A 6-Sprint Approach

This plan breaks the project into manageable 1-week sprints.

### Sprint 0: Setup & Foundation (1-2 days)
*   **Goal:** Create a working, "hello world" version of the full stack.
*   **Tasks:**
    *   Initialize Git repository.
    *   Set up Python virtual environment (`venv`). Create `requirements.txt`.
    *   Set up a barebones FastAPI backend with a single `/` endpoint.
    *   Set up a barebones React frontend using Vite.
    *   Ensure the frontend can successfully make an API call to the backend (handle CORS).

### Sprint 1: Core Backend Logic (Supervised)
*   **Goal:** Create a Python script that perfectly executes the supervised pipeline on a sample dataset (e.g., Telco Churn).
*   **Tasks:**
    *   Write data loading and cleaning functions.
    *   Implement the feature engineering/preprocessing steps.
    *   Train an XGBoost model.
    *   Generate and save SHAP summary plots locally to verify they work.
    *   **Outcome:** A reliable, testable `main_supervised.py` script.

### Sprint 2: Backend API & Basic Frontend Connection
*   **Goal:** Make the supervised analysis accessible via an API endpoint.
*   **Tasks:**
    *   Wrap the logic from Sprint 1 into a `/predict` endpoint in FastAPI.
    *   Build the file upload component and mode selector in React.
    *   On file upload, send the data to the `/predict` endpoint.
    *   Have the backend return a simple JSON response (e.g., `{"status": "success", "accuracy": 0.85}`).
    *   Display this simple response on the frontend.
    *   **Outcome:** A working end-to-end flow, even if the UI is basic.

### Sprint 3: Polished Supervised UI & Visualizations
*   **Goal:** Build the beautiful, interactive dashboard for the supervised results.
*   **Tasks:**
    *   Design the dashboard layout using Ant Design components.
    *   Modify the `/predict` endpoint to return the rich JSON needed for Plotly.
    *   Use Plotly.js in React to render the SHAP Beeswarm plot.
    *   Build the Model Performance card.
    *   Implement the interactive query tool and its corresponding `/query` endpoint.
    *   **Outcome:** A fully functional and polished supervised analysis feature.

### Sprint 4: Unsupervised Backend & API
*   **Goal:** Create the backend logic and API endpoint for the unsupervised path.
*   **Tasks:**
    *   Create a new script, `main_unsupervised.py`, for the unsupervised logic (Elbow Method, K-Means, PCA, profiling).
    *   Wrap this logic into a new `/explore` endpoint in FastAPI.
    *   Test the endpoint thoroughly using FastAPI's built-in docs (`/docs`).
    *   **Outcome:** A fully functional and tested unsupervised backend.

### Sprint 5: Unsupervised UI & Final Polish
*   **Goal:** Build the unsupervised dashboard and integrate all parts of the application.
*   **Tasks:**
    *   Build the `UnsupervisedDashboard` React component.
    *   Use Plotly.js to render the 2D cluster scatter plot.
    *   Display the cluster profile cards.
    *   Add loading spinners for a better user experience during backend processing.
    *   Implement robust error handling (e.g., for incorrect file types, analysis failures).
    *   Write a comprehensive `README.md` (this file!).
    *   **Outcome:** A feature-complete application.

### Sprint 6: Deployment & Documentation
*   **Goal:** Make the application publicly available.
*   **Tasks:**
    *   **Backend:** Containerize the FastAPI application using **Docker**. Deploy it to a service like **Render** or **Heroku**.
    *   **Frontend:** Deploy the static React build to a service like **Netlify** or **Vercel**.
    *   Configure environment variables (e.g., the backend URL for the frontend).
    *   Perform final testing on the live application.
    *   Record a video demo of the project.
    *   **Outcome:** A live, shareable, and well-documented portfolio piece.

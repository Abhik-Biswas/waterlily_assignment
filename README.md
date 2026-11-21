# Diabetes Hospital Readmission Prediction

### Waterlily — Data Science Take-Home Assignment

This repository contains an end-to-end machine learning workflow for analyzing diabetes-related hospital encounters and predicting patient readmission outcomes using the **Diabetes 130-US Hospitals (1999–2008)** dataset.

The project includes exploratory data analysis, data preprocessing, feature engineering, multiclass classification using XGBoost, and a lightweight REST API for model inference.

## 1. Project Objective

Hospital readmissions represent an important challenge for healthcare providers, patients, and insurers. Understanding the factors associated with readmission can help identify higher-risk patients and inform care management strategies.

**Primary objective:** Develop a multiclass classification model to predict whether a patient will be readmitted following a hospital encounter.

The target variable, `readmitted`, has three categories:

| Class | Description |
|---|---|
| `<30` | Readmitted within 30 days |
| `>30` | Readmitted after 30 days |
| `NO` | No recorded readmission |

### Potential Business Applications

- **Healthcare providers:** Identify patients who may benefit from additional discharge planning and follow-up care.
- **Health insurers:** Support population-level risk assessment and targeted care management.
- **Care management teams:** Identify patterns associated with repeat hospitalizations and potential opportunities for intervention.

These are potential applications of the analysis; the model is not intended to replace clinical judgment.

## 2. Dataset

**Source:** [UCI Machine Learning Repository — Diabetes 130-US Hospitals (1999–2008)](https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008)

The dataset contains approximately 100,000 hospital encounters involving patients diagnosed with diabetes across 130 US hospitals between 1999 and 2008.

It includes information about:

- Patient demographics, including age, race, and gender
- Admission type, admission source, and discharge disposition
- Hospital utilization and prior inpatient/outpatient encounters
- Laboratory results, including HbA1c and serum glucose
- Diabetes medications and medication changes
- Diagnoses and other encounter-level characteristics
- Hospital readmission outcomes

The dataset is historical and observational, so its patterns may not generalize directly to current clinical practice.

## 3. Repository Structure

```text
waterlily_assignment/
│
├── Data/
│   ├── diabetic_data.csv
│   ├── diabetes_features.csv
│   └── IDS_mapping.csv
│
├── Take Home Assignment - Water Lily.ipynb
├── app.py
├── xgboost_model.json
├── database.db
├── sample.json
└── README.md
```

### File Descriptions

| File | Description |
|---|---|
| `Data/` | Source data and supporting feature/code mappings |
| `Take Home Assignment - Water Lily.ipynb` | Main analysis, preprocessing, model development, and evaluation notebook |
| `app.py` | FastAPI application exposing health and prediction endpoints |
| `xgboost_model.json` | Serialized trained XGBoost classification model |
| `database.db` | SQLite database containing encounter data |
| `sample.json` | Example JSON payload for testing inference |
| `README.md` | Project documentation |

## 4. Methodology

### Exploratory Data Analysis

The notebook explores patient characteristics, categorical distributions, clinical features, missing values, and their relationships with hospital readmission outcomes.

Particular attention is given to clinically relevant variables such as admission characteristics, discharge disposition, medication usage, and laboratory testing.

### Data Preprocessing and Feature Engineering

The preprocessing workflow includes:

- Examining and handling missing or unknown values
- Exploring categorical and numerical features
- Mapping encoded admission and discharge identifiers to descriptive categories
- Investigating weight categories and missingness
- Preparing categorical features for model training
- Encoding the target variable for multiclass classification

The final XGBoost model uses native categorical feature support rather than requiring all categorical predictors to be one-hot encoded.

### Train / Validation / Test Split

The data is divided using stratified sampling:

| Dataset | Proportion |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

Stratification maintains approximately consistent class distributions across the three subsets.

### Model Development

**Algorithm:** XGBoost Classifier

**Problem type:** Multiclass classification

The model uses:

- Multiclass probability estimation (`multi:softprob`)
- Histogram-based tree construction
- Native categorical feature handling
- Multiclass log loss for evaluation
- Early stopping using validation performance

The notebook also includes a commented-out RandomizedSearchCV implementation for hyperparameter optimization using stratified cross-validation.

### Model Evaluation

The modeling workflow includes:

- Classification report
- Confusion matrix
- Macro-averaged F1 score
- Validation-based early stopping

Macro F1 is particularly useful because the readmission classes are imbalanced. It gives equal weight to each class when summarizing classification performance.

## 5. SQLite Database

The project uses SQLite to store the encounter dataset and support SQL-based exploration.

The database file is:

```text
database.db
```

The notebook loads encounter data into a table named `diabetic_data`.

Example:

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect("database.db")

query = """
SELECT *
FROM diabetic_data
LIMIT 10
"""

result = pd.read_sql_query(query, conn)
print(result)
conn.close()
```

SQLite is embedded and does not require a separate database server.

## 6. REST API

A lightweight inference API is implemented using **FastAPI**.

The application loads the saved XGBoost model and exposes two endpoints.

### GET `/health`

Checks whether the API is running.

**Request:**

```http
GET /health
```

**Response:**

```json
{
  "status": "ok"
}
```

### POST `/predict`

Accepts a JSON object containing the model's expected input features and returns a readmission prediction.

**Request:**

```http
POST /predict
Content-Type: application/json
```

The request body should follow the structure of `sample.json`, containing the feature names expected by the trained model.

**Example response (illustrative):**

```json
{
  "prediction": "NO",
  "prediction_code": 2,
  "probabilities": {
    "<30": 0.12,
    ">30": 0.31,
    "NO": 0.57
  }
}
```

The response contains the predicted class, its encoded numerical value, and estimated probabilities for all three readmission classes.

The API performs basic feature validation and converts categorical inputs to the appropriate data types before inference.

## 7. Installation and Setup

### Prerequisites

- Python 3.10 or later
- pip
- Jupyter Notebook or JupyterLab (for running the analysis)

### Clone the Repository

```bash
git clone https://github.com/Abhik-Biswas/waterlily_assignment.git

cd waterlily_assignment
```

### Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment:

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost \
    matplotlib seaborn plotly \
    fastapi "uvicorn[standard]" \
    pydantic joblib jupyter
```

For reproducibility, use package versions compatible with the environment in which the XGBoost model was trained.

## 8. Running the Application

### Start the API

From the repository root:

```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

The application will be available at:

```text
http://127.0.0.1:8000
```

The API expects `xgboost_model.json` to be present alongside `app.py`.

### Test the Health Endpoint

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok"}
```

### Test the Prediction Endpoint

The repository includes `sample.json`, which can be used as an example inference request.

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d @sample.json
```

The API returns the predicted readmission category and corresponding class probabilities.

### Interactive API Documentation

FastAPI automatically generates interactive documentation at:

**Swagger UI:** http://127.0.0.1:8000/docs

**ReDoc:** http://127.0.0.1:8000/redoc

Both interfaces can be used to inspect the API and its request/response structures.

## 9. Running the Jupyter Notebook

Start Jupyter from the repository root:

```bash
jupyter notebook
```

Open:

```text
Take Home Assignment - Water Lily.ipynb
```

The notebook contains the data analysis and modeling workflow, including data preparation, exploratory analysis, model training, and evaluation.

The source data should be available in the `Data/` directory.

## 10. Limitations and Considerations

Several limitations should be considered when interpreting the results:

**Class imbalance:** The three readmission categories are not equally represented. Overall accuracy alone may not adequately reflect minority-class performance.

**Missing clinical information:** Certain laboratory measurements and patient attributes have substantial missingness. Missing observations may reflect clinical testing practices rather than random data loss.

**Observational data:** Associations between features and readmission do not establish causal relationships.

**Historical dataset:** The data covers 1999–2008 and may not fully represent contemporary clinical workflows or treatment practices.

**Model generalizability:** Performance on a held-out dataset does not guarantee equivalent performance across different hospitals or patient populations.

**Clinical applicability:** Predictions should be interpreted as statistical risk estimates, not medical diagnoses or treatment recommendations.

## 11. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Data processing | pandas, NumPy |
| Database | SQLite |
| Machine learning | XGBoost, scikit-learn |
| Visualization | Matplotlib, Seaborn, Plotly |
| API framework | FastAPI |
| API server | Uvicorn |
| Model serialization | XGBoost JSON |
| Development environment | Jupyter Notebook |

## 12. Future Improvements

Potential extensions include:

- Additional hyperparameter tuning and model comparison
- Model calibration and class-specific threshold optimization
- Feature importance and SHAP-based explainability
- Subgroup performance and fairness analysis
- Stronger API request validation and automated testing
- Containerization using Docker
- Evaluation on more recent and external hospital datasets

---

**Project:** Waterlily — Data Science Take-Home Assignment

**Repository:** [github.com/Abhik-Biswas/waterlily_assignment](https://github.com/Abhik-Biswas/waterlily_assignment)

**Dataset:** UCI Diabetes 130-US Hospitals (1999–2008)

# Diabetes Hospital Readmission Prediction
### Waterlily — Data Science Take-Home Assignment

An end-to-end machine learning project using the **Diabetes 130-US Hospitals (1999–2008)** dataset to analyze hospital encounters and predict patient readmission outcomes.

The project includes exploratory data analysis, feature engineering, multiclass classification using XGBoost, a SQLite database, and a lightweight FastAPI inference service.

## 1. Business Use Cases

### Primary Use Case: Hospital Readmission Prediction

Predict the likelihood of hospital readmission based on a patient's previous encounter.

**Insurance Companies**
- Estimate financial reserves to allocate or release based on readmission risk.
- Develop targeted wellness programs for high-risk patients.

**Hospitals**
- Proactively follow up with patients at elevated risk of readmission.
- Optimize hospital resource allocation.

**Waterlily**
- Help customers understand and plan for potential future readmissions.
- Support financial preparedness for future hospitalization expenses.

### Secondary Use Case 1: Insurance Payer and Procedure Utilization

Analyze whether insurance payer categories influence the number of laboratory and medical procedures performed.

**Government Agencies**
- Monitor healthcare utilization patterns across insurers and providers.
- Identify potentially unnecessary procedures or excessive billing for further investigation.

**Waterlily**
- Provide customized estimates of future healthcare costs based on insurance carriers.
- Explore preventive testing strategies that may help reduce avoidable readmissions.

### Secondary Use Case 2: Hospital Length-of-Stay Prediction

Estimate the duration of a patient's hospitalization based on clinical and admission characteristics.

- Help patients anticipate potential hospitalization-related expenses.
- Estimate time spent in hospitals or healthcare facilities.
- Support future healthcare cost planning and resource allocation.

**Scope:** The primary readmission prediction model is implemented. The secondary use cases are proposed extensions.

## 2. Dataset

**Source:** [UCI Machine Learning Repository — Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008)

The dataset contains approximately 100,000 hospital encounters involving diabetic patients across 130 US hospitals between 1999 and 2008.

Features include:
- Patient demographics (age, race, gender)
- Admission type, source, and discharge disposition
- Diagnoses and prior hospital utilization
- Laboratory results (HbA1c, serum glucose)
- Diabetes medications and treatment changes
- Insurance payer information
- Hospital length of stay

**Target Variable:** `readmitted`

| Class | Description |
|---|---|
| `<30` | Readmitted within 30 days |
| `>30` | Readmitted after 30 days |
| `NO` | No recorded readmission |

## 3. Repository Structure

```text
waterlily_assignment/
│
├── Data/                               # Dataset and mappings
├── Take Home Assignment - Water Lily.ipynb
├── app.py                              # FastAPI inference service
├── database.db                         # SQLite database
├── sample.json                         # Sample prediction request
├── xgboost_model.json                  # Trained XGBoost model
└── README.md
```

## 4. Methodology

**Exploratory Data Analysis**
- Analyzed distributions of patient demographics, clinical characteristics, and readmission outcomes.
- Investigated missing values and categorical feature relationships.
- Explored clinical and operational factors associated with readmission.

**Data Preprocessing**
- Handled missing and unknown categorical values.
- Mapped admission and discharge identifiers to descriptive categories.
- Explored conditional imputation of missing weight categories.
- Prepared categorical features for native XGBoost processing.
- Encoded the multiclass target variable.

**Model Development**
- Algorithm: XGBoost Classifier
- Objective: Multiclass classification (`multi:softprob`)
- Data split: 70% training, 15% validation, 15% testing
- Stratified sampling to preserve class distributions
- Early stopping based on validation log loss

**Model Evaluation**
- Classification report
- Confusion matrix
- Macro-averaged F1 score
- Per-class precision and recall

The notebook also includes a commented-out RandomizedSearchCV implementation for hyperparameter tuning using stratified 5-fold cross-validation.

## 5. SQLite Database

The project uses SQLite for storing and querying encounter-level data.

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect("database.db")

result = pd.read_sql_query(
    "SELECT * FROM diabetic_data LIMIT 10",
    conn
)

print(result)
conn.close()
```

SQLite runs locally without requiring a separate database server.

## 6. REST API

A lightweight REST API is implemented using **FastAPI**, with the trained XGBoost model loaded from `xgboost_model.json`.

### GET `/health`

Checks API availability.

**Response:**
```json
{
  "status": "ok"
}
```

### POST `/predict`

Accepts a JSON payload containing the model's expected features and returns the predicted readmission class and class probabilities.

Use `sample.json` as a reference for the request structure.

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

## 7. Installation and Setup

### Clone the Repository

```bash
git clone https://github.com/Abhik-Biswas/waterlily_assignment.git
cd waterlily_assignment
```

### Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

For Windows:

```bash
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn plotly fastapi "uvicorn[standard]" pydantic joblib jupyter
```

Use an XGBoost version compatible with the serialized model.

## 8. Running the Application

### Start the API

```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

### Test the Health Endpoint

```bash
curl http://127.0.0.1:8000/health
```

Expected output:

```json
{"status": "ok"}
```

### Test the Prediction Endpoint

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d @sample.json
```

### Interactive API Documentation

FastAPI provides automatically generated API documentation:

- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

## 9. Running the Notebook

```bash
jupyter notebook
```

Open `Take Home Assignment - Water Lily.ipynb` to explore the analysis, preprocessing, model development, and evaluation workflow.

The source data is available in the `Data/` directory.

## 10. Limitations and Future Improvements

**Current Limitations**
- Imbalanced readmission classes
- Significant missingness in selected clinical features
- Historical data that may not represent current clinical practices
- Limited generalizability without external validation

**Potential Improvements**
- Hyperparameter optimization and model comparison
- SHAP-based model explainability
- Probability calibration and threshold optimization
- Evaluation across demographic and clinical subgroups
- Additional modeling for payer utilization and hospital length of stay
- API containerization and automated testing

## 11. Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| Data Analysis | pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | XGBoost, scikit-learn |
| Database | SQLite |
| API | FastAPI, Uvicorn |
| Development | Jupyter Notebook |

---

**Project:** Waterlily — Data Science Take-Home Assignment

**Repository:** [github.com/Abhik-Biswas/waterlily_assignment](https://github.com/Abhik-Biswas/waterlily_assignment)

# GridCast

**Short-Term Electricity Demand Forecasting and Explainable Analytics**

GridCast is a data science project for short-term electricity demand forecasting using historical PJM hourly energy consumption data. It combines time-series feature engineering, XGBoost forecasting, model evaluation, feature importance, SHAP explainability, and an interactive Streamlit dashboard.

## Key Features

- Next-hour electricity demand forecasting
- Historical forecast simulation with actual-versus-predicted comparison
- Chronological model evaluation using MAE, RMSE, and R²
- XGBoost feature importance
- SHAP model explainability
- Interactive demand analytics
- Dataset exploration and CSV downloads

## Forecasting Inputs

| Model Feature | Meaning |
| --- | --- |
| Year | Calendar year |
| Month | Calendar month |
| Day | Day of the month |
| Hour | Hour of the day |
| DayOfWeek | Day of the week |
| Lag_1 | Demand one hour earlier |
| Lag_24 | Demand 24 hours earlier |
| Lag_168 | Demand one week earlier |
| Rolling24 | Average demand over the previous 24 hours |
| Rolling168 | Average demand over the previous 7 days |

The application displays human-readable labels while retaining the trained model's original feature names internally.

## Application Pages

### Dashboard
Historical electricity demand, key dataset statistics, monthly demand patterns, and correlation analysis.

### Next-Hour Forecast
Automatically prepares recent historical demand features and forecasts the hour immediately following the latest available observation.

### Historical Forecast Simulator
Tests the model on a selected historical timestamp and compares the forecast with actual recorded demand.

### Model Evaluation
Evaluates the trained XGBoost model on the chronological final 20% of the feature-engineered dataset.

### Feature Importance
Displays feature importance values learned by the XGBoost model.

### SHAP Explainability
Provides global explanations of model behavior using SHAP.

### Analytics
Interactive distribution and box-plot analysis.

### Dataset
Processed dataset preview and CSV download.

## Architecture

```text
PJM Hourly Demand Data
        |
        v
Data Cleaning and Preprocessing
        |
        v
Time-Series Feature Engineering
        |
        v
Model Training and Comparison
        |
        v
Trained XGBoost Model
        |
        +-----------------------+
        |                       |
        v                       v
Next-Hour Forecast       Model Evaluation
        |                       |
        +-----------+-----------+
                    |
                    v
        Explainability & Analytics
                    |
                    v
           Streamlit Dashboard
```

## Technology Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, SHAP, Plotly, Matplotlib, Streamlit, and Joblib.

## Project Structure

```text
GridCast/
├── dashboard/
│   └── app.py
├── data/
│   └── processed/
│       └── clean_energy.csv
├── models/
│   └── xgboost.pkl
├── notebooks/
├── README.md
├── requirements.txt
└── .gitignore
```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## Evaluation Approach

The application uses a chronological split and evaluates the model on the final 20% of the feature-engineered observations. Metrics include MAE, RMSE, and R², along with actual-versus-predicted and residual plots.

## Limitations

GridCast is primarily a short-term forecasting project. The model depends on recent historical demand and should not be presented as a reliable forecasting system for arbitrary dates far beyond the available dataset.

## Future Improvements

- Day-ahead 24-hour forecasting
- Automated ingestion of recent demand data
- Weather and temperature features
- Time-series cross-validation
- Hyperparameter optimization
- Model monitoring and drift detection
- PDF forecast reports

## Author

**Aswanth Thiruchuthan**

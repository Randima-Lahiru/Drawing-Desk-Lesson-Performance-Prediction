# Drawing Desk Lesson Performance Prediction

This MSc machine learning project predicts the expected number of lesson completions within 30 days for Drawing Desk lessons.

## Machine Learning Models

Four regression models are compared:

1. **Extra Trees Regressor** — selected as the new algorithm for the assignment
2. Random Forest Regressor
3. XGBoost Regressor
4. Gradient Boosting Regressor

The three comparison models are used as lecture baselines.

## Target

`Completions_30d`

## Evaluation

The project uses a chronological split:

- 70% training
- 15% validation
- 15% test

Metrics:

- MAE
- RMSE
- R²

The training script tunes each model using `GridSearchCV` with `TimeSeriesSplit`, compares models on the validation set, compares all four models on validation data. **Extra Trees remains the main assignment model** because it is the required new algorithm. It is retrained using training + validation data, evaluated on the final test set, explained using SHAP, and used by the Streamlit dashboard. The strongest lecture baseline is still reported for comparison.

## Explainability

The project includes:

- Feature importance
- SHAP global importance
- SHAP summary plot

## Project Structure

```text
Drawing_Desk_Lesson_Performance_Prediction_Python/
├── data/
│   └── drawing_desk_full_lesson_performance_dataset_updated.csv
├── dashboard/
│   └── app.py
├── models/
│   ├── lesson_performance_model.pkl
│   ├── model_features.pkl
│   └── feature_metadata.pkl
├── outputs/
│   ├── model_comparison.csv
│   ├── final_test_results.csv
│   ├── feature_importance.csv
│   ├── shap_importance.csv
│   ├── shap_summary.png
│   ├── actual_vs_predicted.png
│   └── test_predictions.csv
├── train_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

```bash
cd Drawing_Desk_Lesson_Performance_Prediction_Python
python3.13 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 1. Train the Models

```bash
python train_model.py
```

This command automatically creates the trained model and all evaluation/explainability output files.

## 2. Run the Streamlit Dashboard

```bash
python -m streamlit run dashboard/app.py
```

Open the local URL shown by Streamlit, normally:

```text
http://localhost:8501
```

## Workflow

```text
Dataset
   ↓
Preprocessing
   ↓
Chronological Train / Validation / Test Split
   ↓
Extra Trees + 3 Baseline Models
   ↓
Hyperparameter Tuning
   ↓
Validation Comparison
   ↓
Best Model Selection
   ↓
Final Test Evaluation
   ↓
SHAP + Feature Importance
   ↓
Streamlit Prediction Dashboard
```

## GitHub

https://github.com/Randima-Lahiru/Drawing-Desk-Lesson-Performance-Prediction

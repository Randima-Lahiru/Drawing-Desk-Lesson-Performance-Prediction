from pathlib import Path
import json
import warnings
import shutil

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "drawing_desk_full_lesson_performance_dataset_updated.csv"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "Completions_30d"
RANDOM_STATE = 42


def load_and_prepare_data():
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Convert date fields
    for column in ["publish_date", "campaign_start_date", "campaign_end_date"]:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")

    # Create useful date-based features
    if {"campaign_start_date", "campaign_end_date"}.issubset(df.columns):
        df["campaign_duration_days"] = (
            df["campaign_end_date"] - df["campaign_start_date"]
        ).dt.days

    if {"publish_date", "campaign_start_date"}.issubset(df.columns):
        df["campaign_start_offset_days"] = (
            df["publish_date"] - df["campaign_start_date"]
        ).dt.days

    # Sort chronologically before splitting
    if "publish_date" in df.columns:
        df = df.sort_values("publish_date").reset_index(drop=True)

    # Remove identifiers, leakage, unavailable future outcomes and unused fields
    drop_columns = [
        "lesson_id",
        "lesson_title",
        "course_id",
        "course_title",
        "linked_content_id",
        "category_id",
        "sub_category",
        "publish_date",
        "campaign_start_date",
        "campaign_end_date",
        "event_name",
        "Views_30d",
        "Starts_30d",
        "subscriptions_30d",
        "completion_rate_30d",
        "campaign_impressions",
        "campaign_clicks",
        "paid_spend",
        "course_student_count_current",
        "assumed_fields_generated",
        "publish_date_source",
        "source_lesson_length",
    ]

    X = df.drop(columns=[TARGET] + [c for c in drop_columns if c in df.columns])
    y = pd.to_numeric(df[TARGET], errors="coerce")

    valid_rows = y.notna()
    X = X.loc[valid_rows].reset_index(drop=True)
    y = y.loc[valid_rows].reset_index(drop=True)
    df = df.loc[valid_rows].reset_index(drop=True)

    return df, X, y


def split_data(X, y):
    # Chronological 70 / 15 / 15 split
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train = X.iloc[:train_end].copy()
    y_train = y.iloc[:train_end].copy()
    X_val = X.iloc[train_end:val_end].copy()
    y_val = y.iloc[train_end:val_end].copy()
    X_test = X.iloc[val_end:].copy()
    y_test = y.iloc[val_end:].copy()

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor(X_train):
    # Identify numerical and categorical columns
    numerical_features = X_train.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor, numerical_features, categorical_features


def get_models_and_parameters():
    # Extra Trees is the new algorithm; the other three are lecture baselines
    models = {
        "Extra Trees (New Algorithm)": (
            ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=1),
            {
                "model__n_estimators": [150],
                "model__max_depth": [None, 20],
                "model__min_samples_leaf": [1],
                "model__max_features": ["sqrt", 0.7],
            },
        ),
        "Random Forest": (
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
            {
                "model__n_estimators": [150],
                "model__max_depth": [None],
                "model__min_samples_leaf": [1, 2],
                "model__max_features": [0.7],
            },
        ),
        "XGBoost": (
            XGBRegressor(
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbosity=0,
            ),
            {
                "model__n_estimators": [250],
                "model__max_depth": [3, 5],
                "model__learning_rate": [0.05],
                "model__subsample": [0.8],
                "model__colsample_bytree": [0.8],
            },
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [200, 300],
                "model__learning_rate": [0.05],
                "model__max_depth": [2],
            },
        ),
    }

    return models


def evaluate_predictions(y_true, predictions):
    return {
        "MAE": mean_absolute_error(y_true, predictions),
        "RMSE": root_mean_squared_error(y_true, predictions),
        "R2": r2_score(y_true, predictions),
    }


def tune_and_compare_models(X_train, y_train, X_val, y_val, preprocessor):
    models = get_models_and_parameters()
    time_cv = TimeSeriesSplit(n_splits=3)

    results = []
    fitted_models = {}

    for model_name, (estimator, param_grid) in models.items():
        print(f"\nTraining: {model_name}")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="neg_root_mean_squared_error",
            cv=time_cv,
            n_jobs=1,
            refit=True,
        )

        search.fit(X_train, y_train)
        predictions = search.best_estimator_.predict(X_val)
        metrics = evaluate_predictions(y_val, predictions)

        results.append(
            {
                "Model": model_name,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
                "Best Parameters": json.dumps(search.best_params_),
            }
        )
        fitted_models[model_name] = search.best_estimator_

        print(
            f"Validation -> MAE: {metrics['MAE']:.2f}, "
            f"RMSE: {metrics['RMSE']:.2f}, R2: {metrics['R2']:.3f}"
        )
        print("Best parameters:", search.best_params_)

    comparison_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    return comparison_df, fitted_models


def create_feature_metadata(X, numerical_features, categorical_features):
    metadata = {}

    for column in X.columns:
        if column in numerical_features:
            values = pd.to_numeric(X[column], errors="coerce")
            unique_values = sorted(values.dropna().unique().tolist())
            is_binary = len(unique_values) > 0 and set(unique_values).issubset({0, 1})
            metadata[column] = {
                "type": "numeric",
                "default": float(values.median()) if not values.dropna().empty else 0.0,
                "binary": bool(is_binary),
            }
        else:
            values = X[column].dropna().astype(str)
            mode = values.mode()
            metadata[column] = {
                "type": "categorical",
                "default": str(mode.iloc[0]) if not mode.empty else "Unknown",
                "options": sorted(values.unique().tolist()),
                "binary": False,
            }

    return metadata


def save_feature_importance(final_model):
    preprocessor = final_model.named_steps["preprocessor"]
    estimator = final_model.named_steps["model"]

    transformed_names = preprocessor.get_feature_names_out()
    clean_names = [
        name.replace("num__", "").replace("cat__", "")
        for name in transformed_names
    ]

    if hasattr(estimator, "feature_importances_"):
        importance = estimator.feature_importances_
        importance_df = pd.DataFrame(
            {"Feature": clean_names, "Importance": importance}
        ).sort_values("Importance", ascending=False)
        importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
        return importance_df

    return pd.DataFrame()


def save_shap_importance(final_model, X_test):
    preprocessor = final_model.named_steps["preprocessor"]
    estimator = final_model.named_steps["model"]

    transformed = preprocessor.transform(X_test)
    feature_names = [
        name.replace("num__", "").replace("cat__", "")
        for name in preprocessor.get_feature_names_out()
    ]

    # Limit SHAP evaluation to keep the script practical on a laptop
    sample_size = min(150, len(X_test))
    transformed_sample = transformed[:sample_size]

    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(transformed_sample)
    shap_array = np.asarray(shap_values)

    if shap_array.ndim == 3:
        shap_array = shap_array[..., 0]

    mean_abs_shap = np.abs(shap_array).mean(axis=0)
    shap_df = pd.DataFrame(
        {"Feature": feature_names, "Mean_Absolute_SHAP": mean_abs_shap}
    ).sort_values("Mean_Absolute_SHAP", ascending=False)
    shap_df.to_csv(OUTPUT_DIR / "shap_importance.csv", index=False)

    # Save a SHAP summary plot for the report/presentation
    plt.figure()
    shap.summary_plot(
        shap_array,
        transformed_sample,
        feature_names=feature_names,
        show=False,
        max_display=15,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_summary.png", dpi=180, bbox_inches="tight")
    plt.close()

    return shap_df


def save_evaluation_plots(y_test, predictions):
    # Actual vs predicted
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, predictions, alpha=0.65)
    low = min(float(y_test.min()), float(predictions.min()))
    high = max(float(y_test.max()), float(predictions.max()))
    plt.plot([low, high], [low, high], linestyle="--")
    plt.xlabel("Actual Completions")
    plt.ylabel("Predicted Completions")
    plt.title("Actual vs Predicted 30-Day Completions")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=180)
    plt.close()


def main():
    print("Drawing Desk Lesson Performance Prediction")
    print("=" * 48)

    df, X, y = load_and_prepare_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    print(f"Dataset rows: {len(X)}")
    print(f"Features used: {X.shape[1]}")
    print(f"Train: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")

    preprocessor, numerical_features, categorical_features = build_preprocessor(X_train)

    comparison_df, fitted_models = tune_and_compare_models(
        X_train, y_train, X_val, y_val, preprocessor
    )

    comparison_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    dashboard_dir = PROJECT_DIR / "dashboard"
    dashboard_dir.mkdir(exist_ok=True)
    comparison_df.to_csv(dashboard_dir / "model_comparison.csv", index=False)
    print("\nValidation model comparison:")
    print(comparison_df[["Model", "MAE", "RMSE", "R2"]].to_string(index=False))

    # Extra Trees is the main assignment model because it is the new algorithm.
    # The other three models are lecture baselines used for comparison.
    assignment_model_name = "Extra Trees (New Algorithm)"
    best_validation_model_name = comparison_df.iloc[0]["Model"]
    final_model = fitted_models[assignment_model_name]

    print(f"\nAssignment model: {assignment_model_name}")
    print(f"Best validation model for comparison: {best_validation_model_name}")

    # Retrain the assignment model on training + validation data
    X_train_final = pd.concat([X_train, X_val], axis=0)
    y_train_final = pd.concat([y_train, y_val], axis=0)
    final_model.fit(X_train_final, y_train_final)

    # Final test evaluation of the new algorithm
    test_predictions = final_model.predict(X_test)
    final_metrics = evaluate_predictions(y_test, test_predictions)

    final_results = pd.DataFrame(
        [
            {
                "Assignment Model": assignment_model_name,
                "Best Validation Model": best_validation_model_name,
                "Test MAE": final_metrics["MAE"],
                "Test RMSE": final_metrics["RMSE"],
                "Test R2": final_metrics["R2"],
            }
        ]
    )
    final_results.to_csv(OUTPUT_DIR / "final_test_results.csv", index=False)

    predictions_df = pd.DataFrame(
        {
            "Actual_Completions_30d": y_test.to_numpy(),
            "Predicted_Completions_30d": test_predictions,
            "Absolute_Error": np.abs(y_test.to_numpy() - test_predictions),
        }
    )
    predictions_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    # Save model and feature information
    joblib.dump(final_model, MODEL_DIR / "lesson_performance_model.pkl")
    joblib.dump(X.columns.tolist(), MODEL_DIR / "model_features.pkl")

    feature_metadata = create_feature_metadata(
        X, numerical_features, categorical_features
    )
    joblib.dump(feature_metadata, MODEL_DIR / "feature_metadata.pkl")

    # Save explainability outputs
    importance_df = save_feature_importance(final_model)
    if not importance_df.empty:
        importance_df.to_csv(PROJECT_DIR / "dashboard" / "feature_importance.csv", index=False)
    shap_df = save_shap_importance(final_model, X_test)
    save_evaluation_plots(y_test, test_predictions)

    print("\nFinal test results:")
    print(f"MAE : {final_metrics['MAE']:.2f}")
    print(f"RMSE: {final_metrics['RMSE']:.2f}")
    print(f"R2  : {final_metrics['R2']:.3f}")

    if not importance_df.empty:
        print("\nTop 10 feature importances:")
        print(importance_df.head(10).to_string(index=False))

    print("\nTop 10 SHAP features:")
    print(shap_df.head(10).to_string(index=False))

    print("\nGenerated files:")
    print("models/lesson_performance_model.pkl")
    print("models/model_features.pkl")
    print("models/feature_metadata.pkl")
    print("outputs/model_comparison.csv")
    print("outputs/final_test_results.csv")
    print("outputs/feature_importance.csv")
    print("outputs/shap_importance.csv")
    print("outputs/shap_summary.png")
    print("outputs/actual_vs_predicted.png")


if __name__ == "__main__":
    main()

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent

MODEL_PATH = PROJECT_DIR / "models" / "lesson_performance_model.pkl"
FEATURES_PATH = PROJECT_DIR / "models" / "model_features.pkl"
METADATA_PATH = PROJECT_DIR / "models" / "feature_metadata.pkl"
DATA_PATH = PROJECT_DIR / "data" / "drawing_desk_full_lesson_performance_dataset_updated.csv"
COMPARISON_PATH = PROJECT_DIR / "outputs" / "model_comparison.csv"
FINAL_RESULTS_PATH = PROJECT_DIR / "outputs" / "final_test_results.csv"
IMPORTANCE_PATH = PROJECT_DIR / "outputs" / "feature_importance.csv"
SHAP_PATH = PROJECT_DIR / "outputs" / "shap_importance.csv"
ACTUAL_PREDICTED_PLOT = PROJECT_DIR / "outputs" / "actual_vs_predicted.png"
SHAP_SUMMARY_PLOT = PROJECT_DIR / "outputs" / "shap_summary.png"

st.set_page_config(
    page_title="Drawing Desk Lesson Performance Prediction",
    page_icon="🎨",
    layout="wide",
)


@st.cache_resource
def load_model_files():
    model = joblib.load(MODEL_PATH)
    model_features = joblib.load(FEATURES_PATH)
    metadata = joblib.load(METADATA_PATH)
    return model, model_features, metadata


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


def pretty_label(name):
    replacements = {
        "30d": "30-Day",
        "is": "Is",
    }
    label = name.replace("_", " ").title()
    for old, new in replacements.items():
        label = label.replace(old.title(), new)
    return label


def create_input(feature, metadata):
    details = metadata[feature]
    label = pretty_label(feature)

    if details["type"] == "numeric":
        if details.get("binary", False):
            default_index = 1 if int(round(details["default"])) == 1 else 0
            return st.selectbox(
                label,
                [0, 1],
                index=default_index,
                format_func=lambda value: "Yes" if value == 1 else "No",
                key=feature,
            )

        default_value = float(details.get("default", 0.0))
        return st.number_input(label, value=default_value, key=feature)

    options = details.get("options", [])
    default = details.get("default", "Unknown")

    if options:
        default_index = options.index(default) if default in options else 0
        return st.selectbox(label, options, index=default_index, key=feature)

    return st.text_input(label, value=default, key=feature)


def group_features(model_features):
    lesson_keywords = {
        "category", "difficulty", "target_age_group", "art_style", "subject_type",
        "estimated_duration_minutes", "step_count", "has_video", "has_voice_guidance",
        "has_reference_image", "is_beginner_friendly", "is_kids_focused",
        "is_trending_style", "is_seasonal", "target_event", "days_before_event",
        "is_free", "lesson_thumbnail_style", "thumbnail_has_character",
        "thumbnail_has_text", "lesson_complexity_score", "course_lesson_count",
        "lesson_position_in_course", "category_order", "idea_source"
    }

    promotion_keywords = {
        "is_run_marketing_campaign", "campaign_type", "is_send_push_notification",
        "is_run_social_media_campaign", "is_run_email_campaign", "in_app_banner_used",
        "homepage_featured", "featured_position", "promotion_country", "target_audience",
        "category_featured", "featured_days_count", "publish_month", "publish_weekday",
        "publish_hour", "Season", "campaign_duration_days", "campaign_start_offset_days",
        "lesson_list_position", "course_order_index"
    }

    groups = {
        "Lesson Details": [],
        "Promotion & Release Details": [],
        "Historical Context": [],
    }

    for feature in model_features:
        if feature in lesson_keywords:
            groups["Lesson Details"].append(feature)
        elif feature in promotion_keywords:
            groups["Promotion & Release Details"].append(feature)
        else:
            groups["Historical Context"].append(feature)

    return groups


st.title("🎨 Drawing Desk Lesson Performance Prediction")
st.caption("Predict expected lesson completions within 30 days and review model performance.")

if not MODEL_PATH.exists():
    st.error("Model files were not found. Run `python train_model.py` first.")
    st.stop()

model, model_features, feature_metadata = load_model_files()
df = load_dataset()

prediction_tab, comparison_tab, explanation_tab, project_tab = st.tabs(
    ["Prediction", "Model Evaluation", "Explainability", "Project Information"]
)


with prediction_tab:
    st.subheader("New Lesson Prediction")
    st.write("Enter the planned lesson details and predict expected 30-day completions.")

    input_values = {}
    groups = group_features(model_features)

    for group_name, features in groups.items():
        if not features:
            continue

        with st.expander(group_name, expanded=True):
            col1, col2 = st.columns(2)
            for index, feature in enumerate(features):
                target_column = col1 if index % 2 == 0 else col2
                with target_column:
                    input_values[feature] = create_input(feature, feature_metadata)

    if st.button("Predict 30-Day Completions", type="primary"):
        input_df = pd.DataFrame([input_values])
        input_df = input_df[model_features]

        prediction = float(model.predict(input_df)[0])
        prediction = max(0.0, prediction)

        st.success(f"Predicted 30-Day Completions: {round(prediction):,}")

        q1 = df["Completions_30d"].quantile(0.25)
        q3 = df["Completions_30d"].quantile(0.75)

        if prediction < q1:
            level = "Low"
        elif prediction > q3:
            level = "High"
        else:
            level = "Medium"

        col1, col2 = st.columns(2)
        col1.metric("Predicted Completions", f"{round(prediction):,}")
        col2.metric("Expected Performance Level", level)


with comparison_tab:
    st.subheader("Model Evaluation")

    if COMPARISON_PATH.exists():
        comparison_df = pd.read_csv(COMPARISON_PATH)
        display_df = comparison_df[["Model", "MAE", "RMSE", "R2"]].copy()
        display_df[["MAE", "RMSE", "R2"]] = display_df[["MAE", "RMSE", "R2"]].round(3)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        chart_df = comparison_df.set_index("Model")[["RMSE"]]
        st.bar_chart(chart_df)

    if FINAL_RESULTS_PATH.exists():
        st.subheader("Final Test Performance")
        final_df = pd.read_csv(FINAL_RESULTS_PATH)
        row = final_df.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Test MAE", f"{row['Test MAE']:.2f}")
        c2.metric("Test RMSE", f"{row['Test RMSE']:.2f}")
        c3.metric("Test R²", f"{row['Test R2']:.3f}")
        st.caption(f"Assignment model: {row['Assignment Model']}")
        st.caption(f"Best validation comparison model: {row['Best Validation Model']}")

    if ACTUAL_PREDICTED_PLOT.exists():
        st.image(str(ACTUAL_PREDICTED_PLOT), caption="Actual vs Predicted Test Results")


with explanation_tab:
    st.subheader("Model Explainability")
    st.write("Feature importance and SHAP are used to understand which inputs influence predictions.")

    if IMPORTANCE_PATH.exists():
        importance_df = pd.read_csv(IMPORTANCE_PATH).head(15)
        st.markdown("#### Top Feature Importances")
        st.dataframe(importance_df, use_container_width=True, hide_index=True)
        st.bar_chart(importance_df.set_index("Feature")[["Importance"]])

    if SHAP_PATH.exists():
        shap_df = pd.read_csv(SHAP_PATH).head(15)
        st.markdown("#### SHAP Global Importance")
        st.dataframe(shap_df, use_container_width=True, hide_index=True)

    if SHAP_SUMMARY_PLOT.exists():
        st.image(str(SHAP_SUMMARY_PLOT), caption="SHAP Summary Plot")


with project_tab:
    st.subheader("Project Information")
    st.markdown(
        """
        **Prediction target:** `Completions_30d`  
        **Task:** Regression  
        **Data split:** Chronological 70% training / 15% validation / 15% test  
        **New algorithm:** Extra Trees Regressor  
        **Lecture comparison models:** Random Forest, XGBoost, Gradient Boosting  
        **Evaluation metrics:** MAE, RMSE, R²  
        **Explainability:** Feature Importance and SHAP
        """
    )

    st.info(
        "The model is intended to support lesson planning and promotion decisions. "
        "Predictions should be used together with product and domain knowledge."
    )

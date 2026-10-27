# Drawing Desk Lesson Performance Prediction UI

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# Project paths
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent

MODEL_PATH = PROJECT_DIR / "models" / "lesson_performance_model.pkl"
FEATURES_PATH = PROJECT_DIR / "models" / "model_features.pkl"
DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "drawing_desk_full_lesson_performance_dataset_updated.csv"
)


# Page configuration
st.set_page_config(
    page_title="Drawing Desk Lesson Performance Prediction",
    page_icon="🎨",
    layout="wide",
)


# Load model
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    model_features = joblib.load(FEATURES_PATH)

    return model, model_features


# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


model, model_features = load_model()
df = load_data()


# Read numerical and categorical columns from the trained pipeline
preprocessor = model.named_steps["preprocessor"]

numerical_features = []
categorical_features = []

for name, transformer, columns in preprocessor.transformers_:
    if name == "num":
        numerical_features = list(columns)

    elif name == "cat":
        categorical_features = list(columns)


# Get available categorical values from dataset
def get_options(column):
    if column not in df.columns:
        return []

    values = (
        df[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    return sorted(values)


# Check whether a feature is binary
def is_binary_feature(column):
    if column not in df.columns:
        return False

    values = (
        pd.to_numeric(
            df[column],
            errors="coerce"
        )
        .dropna()
        .unique()
        .tolist()
    )

    return len(values) > 0 and set(values).issubset({0, 1})


# Create Streamlit input based on trained feature type
def create_input(column):
    label = column.replace("_", " ").title()

    if is_binary_feature(column):
        return st.selectbox(
            label,
            [0, 1],
            format_func=lambda value: "Yes" if value == 1 else "No",
            key=column,
        )

    if column in numerical_features:
        default_value = 0.0

        if column in df.columns:
            numeric_values = pd.to_numeric(
                df[column],
                errors="coerce",
            )

            if not numeric_values.dropna().empty:
                default_value = float(
                    numeric_values.median()
                )

        return st.number_input(
            label,
            value=default_value,
            key=column,
        )

    if column in categorical_features:
        options = get_options(column)

        if options:
            return st.selectbox(
                label,
                options,
                key=column,
            )

        return st.text_input(
            label,
            value="Unknown",
            key=column,
        )

    return st.text_input(
        label,
        key=column,
    )


# Dashboard header
st.title("🎨 Drawing Desk Lesson Performance Prediction")

st.write(
    "Enter the planned lesson details below to estimate "
    "the expected number of lesson completions within 30 days."
)

st.divider()


# Prediction form
input_values = {}

total_features = len(model_features)
group_size = max(
    1,
    (total_features + 2) // 3,
)

feature_groups = [
    model_features[:group_size],
    model_features[group_size:group_size * 2],
    model_features[group_size * 2:],
]

group_titles = [
    "Lesson Details",
    "Promotion & Release Details",
    "Historical Context",
]


for title, feature_group in zip(
    group_titles,
    feature_groups,
):
    if not feature_group:
        continue

    with st.expander(
        title,
        expanded=True,
    ):
        col1, col2 = st.columns(2)

        for index, feature in enumerate(
            feature_group
        ):
            selected_column = (
                col1
                if index % 2 == 0
                else col2
            )

            with selected_column:
                input_values[feature] = (
                    create_input(feature)
                )


st.divider()


# Prediction
if st.button(
    "Predict 30-Day Completions",
    type="primary",
    use_container_width=True,
):
    input_df = pd.DataFrame(
        [input_values]
    )

    # Ensure numerical fields are numeric
    for column in numerical_features:
        if column in input_df.columns:
            input_df[column] = pd.to_numeric(
                input_df[column],
                errors="coerce",
            )

    # Keep exact training feature order
    input_df = input_df[
        model_features
    ]

    prediction = model.predict(
        input_df
    )[0]

    prediction = max(
        0,
        round(float(prediction)),
    )

    st.success(
        f"Predicted 30-Day Completions: "
        f"{prediction:,}"
    )

    # Performance level based on dataset distribution
    if "Completions_30d" in df.columns:
        q1 = df[
            "Completions_30d"
        ].quantile(0.25)

        q3 = df[
            "Completions_30d"
        ].quantile(0.75)

        if prediction < q1:
            performance_level = "Low"

        elif prediction > q3:
            performance_level = "High"

        else:
            performance_level = "Medium"

        st.metric(
            "Expected Performance Level",
            performance_level,
        )

    st.caption(
        "This prediction is intended to support "
        "lesson planning and promotion decisions."
    )

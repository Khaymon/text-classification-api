import streamlit as st
import requests
import pandas as pd

import os

# Set the base URL for the API
API_BASE_URL = f"http://{os.environ.get('API_HOST', 'localhost')}:{os.environ.get('API_PORT', '8005')}"


# Function to fetch datasets from the API
def get_datasets():
    try:
        response = requests.get(f"{API_BASE_URL}/datasets")
        response.raise_for_status()
        return response.json().get("datasets", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching datasets: {e}")
        return []


# Function to fetch models from the API
def get_models():
    try:
        response = requests.get(f"{API_BASE_URL}/models")
        response.raise_for_status()
        return response.json().get("models", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching models: {e}")
        return []


# Function to fetch model configurations from the API
def get_model_configs():
    try:
        response = requests.get(f"{API_BASE_URL}/models/configs")
        response.raise_for_status()
        return response.json().get("models", {})
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching model configurations: {e}")
        return {}


# Function to fetch trained models (artifacts) from the API
def get_trained_models():
    try:
        response = requests.get(f"{API_BASE_URL}/models/artifacts")
        response.raise_for_status()
        return response.json().get("artifacts", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching trained models: {e}")
        return []


def upload_dataset_page():
    st.title("Upload dataset")
    dataset_name = st.text_area("Dataset name")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        dataset = pd.read_csv(uploaded_file)
        data = [(row.text, row.target) for _, row in dataset.iterrows()]
        upload_dataset_request = {"name": dataset_name, "data": data}

        if st.button("Upload dataset"):
            try:
                with st.spinner("Uploading the dataset..."):
                    response = requests.post(
                        f"{API_BASE_URL}/datasets/upload", json=upload_dataset_request
                    )
                    response.raise_for_status()
                    result = response.json()
                    st.write("**Message:**", result.get("message"))
            except requests.exceptions.RequestException as e:
                st.error(f"Error during training: {e}")


def train_page():
    st.title("Train Machine Learning Models")

    datasets = get_datasets()
    if not datasets:
        st.warning("No datasets available for training.")
        return

    models = get_models()
    model = st.selectbox("Select Model", models)

    dataset = st.selectbox("Select Dataset", datasets)

    st.subheader("Model Parameters")

    params = {}

    if model == "logistic_regression":
        params["C"] = st.number_input(
            "Regularization Strength (C)", min_value=0.01, value=1.0
        )
        params["max_iter"] = st.slider(
            "Maximum Iterations", min_value=100, max_value=1000, value=100
        )
    elif model == "catboost":
        params["iterations"] = st.slider(
            "Iterations", min_value=100, max_value=1000, value=500
        )
        params["depth"] = st.slider("Depth", min_value=3, max_value=10, value=6)
        params["learning_rate"] = st.number_input(
            "Learning Rate", min_value=0.01, value=0.1
        )

    if st.button("Train Model"):
        train_request = {
            "dataset_name": dataset,
            "model": {
                "name": model,
                "configuration": {
                    "preprocessor": {
                        "preprocessors": [
                            {"name": "tfidf", "params": {}},
                            {"name": "drop", "params": {"columns": ["text"]}},
                        ]
                    },
                    "model_configuration": params,
                },
            },
        }

        try:
            with st.spinner("Training the model..."):
                response = requests.post(
                    f"{API_BASE_URL}/models/train", json=train_request
                )
                response.raise_for_status()
                result = response.json()
                st.success("Model trained successfully!")
                st.write("**Model Name:**", result.get("model_name"))
                st.write("**Metrics:**", result.get("metrics"))
        except requests.exceptions.RequestException as e:
            st.error(f"Error during training: {e}")


def predict_page():
    st.title("Predict with Trained Models")

    trained_models = get_trained_models()
    if not trained_models:
        st.warning("No trained models available for prediction.")
        return

    model = st.selectbox("Select Trained Model", trained_models)

    input_text = st.text_area("Enter Text for Prediction", height=200)

    if st.button("Predict"):
        if not input_text.strip():
            st.error("Please enter text for prediction.")
            return

        predict_request = {
            "model_artifact_name": model,
            "data": [input_text],
        }

        try:
            with st.spinner("Making prediction..."):
                response = requests.post(
                    f"{API_BASE_URL}/models/predict", json=predict_request
                )
                response.raise_for_status()
                result = response.json()
                st.success("Prediction successful!")
                st.write("**Prediction:**", result.get("predictions")[0])
        except requests.exceptions.RequestException as e:
            st.error(f"Error during prediction: {e}")


def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Train", "Predict", "Datasets"])

    if page == "Train":
        train_page()
    elif page == "Predict":
        predict_page()
    elif page == "Datasets":
        upload_dataset_page()


if __name__ == "__main__":
    main()

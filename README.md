Symptom-Based Disease Prediction SystemThis repository contains a Machine Learning pipeline designed to predict potential diseases based on user-reported symptoms. The model is built using a Random Forest Classifier and is optimized for integration into web applications.🚀 OverviewThe system analyzes 132 unique symptoms to provide the top three most likely diagnoses along with confidence percentages. It includes pre-processed datasets, training scripts, and exported model artifacts for seamless deployment.Key FeaturesMulti-Disease Prediction: Capable of identifying 41 unique diseases.Probabilistic Output: Returns the top 3 potential matches instead of a single result.Deployment Ready: Includes exported .pkl and .json files for backend integration.🛠️ Tech StackLanguage: PythonLibraries: Pandas, NumPy, Scikit-learn, JoblibModel: Random Forest Classifier ($n=100$)📦 Project StructuresymptonML.py: Core script for data loading, training, and evaluation.symptom_model.pkl: The trained AI "brain".symptom_encoder.pkl: Dictionary to translate numerical predictions back to disease names.symptom_features.json: A standard list of 132 symptoms required for input.🔌 Integration Guide (For Web Developers)1. Input FormatThe model requires a binary vector of length 132.1: Symptom is present.0: Symptom is absent.2. Implementation Example (Flask/FastAPI)To use the model in your web app, load the artifacts and map user selections to the feature vector:Pythonimport joblib
import numpy as np

# Load the model and features
model = joblib.load('symptom_model.pkl')
encoder = joblib.load('symptom_encoder.pkl')

def get_prediction(user_input_list):
    # input_vector must be 132 elements long
    # matching the order in symptom_features.json
    prediction = model.predict_proba([input_vector])[0]
    # ... process top results

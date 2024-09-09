from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import google.generativeai as genai
import time
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

MODEL_FILE = 'gb_classifier.joblib'
FEATURE_NAMES_FILE = 'feature_names.joblib'

def train_model():
    df = pd.read_csv('a.csv')
    df['risk_score'] = df['risk_of_escape'].astype(int) + df['risk_of_influence'].astype(int) + df['served_half_term'].astype(int)
    df['penalty_severity'] = df['penalty'].map({"Fine": 1, "Imprisonment": 2, "Both": 3}) * df['imprisonment_duration_served']
    X = df.drop(["bail_eligibility", "case_id", "penalty", "imprisonment_duration_served", "risk_of_escape", "risk_of_influence", "served_half_term"], axis=1)
    y = df["bail_eligibility"]
    X = pd.get_dummies(X, drop_first=True)
    joblib.dump(X.columns.tolist(), FEATURE_NAMES_FILE)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = GradientBoostingClassifier(random_state=42, n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    joblib.dump(model, MODEL_FILE)
    return model, X.columns.tolist()

def load_or_train_model():
    if os.path.exists(MODEL_FILE) and os.path.exists(FEATURE_NAMES_FILE):
        try:
            model = joblib.load(MODEL_FILE)
            feature_names = joblib.load(FEATURE_NAMES_FILE)
            return model, feature_names
        except:
            print("Error loading model or feature names. Training a new one.")
    return train_model()

model, feature_names = load_or_train_model()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_df = pd.DataFrame([data])
    input_df['risk_score'] = (input_df['risk_of_escape'].astype(int) +
                            input_df['risk_of_influence'].astype(int) +
                            input_df['served_half_term'].astype(int))
    input_df['penalty_severity'] = input_df['penalty'].map({"Fine": 1, "Imprisonment": 2, "Both": 3}) * input_df['imprisonment_duration_served']
    input_df = input_df.drop(["penalty", "imprisonment_duration_served", "risk_of_escape", "risk_of_influence", "served_half_term"], axis=1)
    input_df = pd.get_dummies(input_df, drop_first=True)
    input_df = input_df.reindex(columns=feature_names, fill_value=0)
    prediction = model.predict(input_df)
    return jsonify({'bail_eligibility': bool(prediction[0])})

def get_bail_application(lawyer_name: str, input_text: str, n_words: int, theme: str) -> str:
    try:
        app.logger.info("Initializing Gemini LLM model...")
        start_time = time.time()
        prompt = f"""
        [Your Name] = {input_text}
        [Phone Number] 7/9/24
        Honorable Judge Harshit Kolkata High Court Kolkata -70002
        "write a bail application.
        name of lawyer will be {lawyer_name}
        name of client will be {input_text}
        crime will be {theme}
        number of word in the bail application will be {n_words} words.
        city will be kolkata
        and judge name will be raghav jain
        """
        app.logger.info("Generating bail application...")
        response = model.generate_content(prompt)
        app.logger.info(f"Bail application generated in {time.time() - start_time:.2f} seconds")
        return response.text
    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return None

@app.route('/api/generate_bail_application', methods=['POST'])
def generate_bail_application():
    try:
        data = request.json
        lawyer_name = data.get('lawyer_name')
        input_text = data.get('input_text')
        n_words = data.get('n_words', 200)
        theme = data.get('theme')

        if not lawyer_name or not input_text:
            return jsonify({"success": False, "error": "Missing required fields"}), 400

        if not isinstance(n_words, int) or n_words <= 0:
            return jsonify({"success": False, "error": "Invalid n_words value"}), 400

        response = get_bail_application(lawyer_name, input_text, n_words, theme)

        if response:
            return jsonify({"success": True, "bail_application": response})
        else:
            return jsonify({"success": False, "error": "Failed to generate bail application"}), 500
    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel('gemini-pro')
    app.run(debug=True, host='0.0.0.0', port=5000)

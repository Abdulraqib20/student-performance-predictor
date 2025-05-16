from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import joblib
import numpy as np
import os
import pandas as pd # Added for CSV parsing if needed later, not strictly used for mappings now
import random # Added for random value generation
from functools import wraps # Added for login_required decorator

app = Flask(__name__)
# It's crucial to set a secret key for session management.
# In a real application, use a strong, randomly generated key and keep it secret.
# For example, you can generate one using: import os; os.urandom(24)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/' # Example key, replace in production

# Configuration for login
USERNAME = "21/52HL155"
PASSWORD = "hayzed"

# More robust path to the models directory
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_ROOT, '..', 'models')

# Define all feature names based on student-data.csv (excluding 'passed')
ALL_FEATURE_NAMES = [
    'school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
    'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures',
    'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet',
    'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences'
]

FEATURE_DESCRIPTIONS = {
    'school': "Student's school (GP or MS)",
    'sex': "Student's sex (F - female or M - male)",
    'age': "Student's age (from 15 to 22)",
    'address': "Student's home address type (U - urban or R - rural)",
    'famsize': "Family size (LE3 - less or equal to 3 or GT3 - greater than 3)",
    'Pstatus': "Parent's cohabitation status (T - living together or A - apart)",
    'Medu': "Mother's education (0 - none, 1 - primary (4th grade), 2 - 5th to 9th grade, 3 - secondary, 4 - higher education)",
    'Fedu': "Father's education (0 - none, 1 - primary (4th grade), 2 - 5th to 9th grade, 3 - secondary, 4 - higher education)",
    'Mjob': "Mother's job (teacher, health care, services, at_home, or other)",
    'Fjob': "Father's job (teacher, health care, services, at_home, or other)",
    'reason': "Reason to choose this school (home, reputation, course preference, or other)",
    'guardian': "Student's guardian (mother, father, or other)",
    'traveltime': "Home to school travel time (1: <15 min, 2: 15-30 min, 3: 30min-1hr, 4: >1hr)",
    'studytime': "Weekly study time (1: <2 hrs, 2: 2-5 hrs, 3: 5-10 hrs, 4: >10 hrs)",
    'failures': "Number of past class failures (n if 1<=n<3, else 4)",
    'schoolsup': "Extra educational support (yes or no)",
    'famsup': "Family educational support (yes or no)",
    'paid': "Extra paid classes within the course subject (yes or no)",
    'activities': "Extra-curricular activities (yes or no)",
    'nursery': "Attended nursery school (yes or no)",
    'higher': "Wants to take higher education (yes or no)",
    'internet': "Internet access at home (yes or no)",
    'romantic': "With a romantic relationship (yes or no)",
    'famrel': "Quality of family relationships (1 - very bad to 5 - excellent)",
    'freetime': "Free time after school (1 - very low to 5 - very high)",
    'goout': "Going out with friends (1 - very low to 5 - very high)",
    'Dalc': "Workday alcohol consumption (1 - very low to 5 - very high)",
    'Walc': "Weekend alcohol consumption (1 - very low to 5 - very high)",
    'health': "Current health status (1 - very bad to 5 - very good)",
    'absences': "Number of school absences (0 to 93)"
}

# Define categorical features and their expected unique values (derived from CSV)
# IMPORTANT: This manual encoding might differ from your notebook's LabelEncoder.
# For production, save/load your encoders.
CATEGORICAL_FEATURES_MAP = {
    'school': {'GP': 0, 'MS': 1},
    'sex': {'F': 0, 'M': 1},
    'address': {'U': 0, 'R': 1},
    'famsize': {'GT3': 0, 'LE3': 1}, # GT3: Greater than 3, LE3: Less or Equal to 3
    'Pstatus': {'A': 0, 'T': 1},     # A: Apart, T: Together
    'Mjob': {'at_home': 0, 'health': 1, 'other': 2, 'services': 3, 'teacher': 4},
    'Fjob': {'at_home': 0, 'health': 1, 'other': 2, 'services': 3, 'teacher': 4},
    'reason': {'course': 0, 'home': 1, 'other': 2, 'reputation': 3},
    'guardian': {'father': 0, 'mother': 1, 'other': 2},
    'schoolsup': {'no': 0, 'yes': 1}, # Consistent with typical LabelEncoder (no=0, yes=1)
    'famsup': {'no': 0, 'yes': 1},
    'paid': {'no': 0, 'yes': 1},
    'activities': {'no': 0, 'yes': 1},
    'nursery': {'no': 0, 'yes': 1},
    'higher': {'no': 0, 'yes': 1}, # Assuming 'no' should be 0, 'yes' should be 1 for consistency
    'internet': {'no': 0, 'yes': 1},
    'romantic': {'no': 0, 'yes': 1}
}

# Numerical features are the ones not in CATEGORICAL_FEATURES_MAP keys
NUMERICAL_FEATURES = [f for f in ALL_FEATURE_NAMES if f not in CATEGORICAL_FEATURES_MAP]

# Helper for random generation
FEATURE_RANGES = {
    'age': (15, 22),
    'Medu': (0, 4),
    'Fedu': (0, 4),
    'traveltime': (1, 4),
    'studytime': (1, 4),
    'failures': (0, 3), # User inputs 0,1,2,3. Model might treat 3 as '3 or more'.
    'famrel': (1, 5),
    'freetime': (1, 5),
    'goout': (1, 5),
    'Dalc': (1, 5),
    'Walc': (1, 5),
    'health': (1, 5),
    'absences': (0, 93)
}

# Dictionary to map model filenames to more descriptive names
MODEL_DISPLAY_NAMES = {
    'lr_model.joblib': 'Logistic Regression',
    'rf_model.joblib': 'Random Forest Classifier',
    'knn_model.joblib': 'K-Nearest Neighbors (KNN)',
    'svm_model.joblib': 'Support Vector Machine (SVM)',
    'xgb_model.joblib': 'XGBoost Classifier'
}

MODEL_PERFORMANCE_METRICS = {
    'lr_model.joblib': {
        'name': 'Logistic Regression',
        'Accuracy': '67.09%', 'F1 Score': '77.59%', 'Precision': '71.43%', 'Recall': '84.91%', 'ROC-AUC': '60.09%'
    },
    'xgb_model.joblib': {
        'name': 'XGBoost Classifier',
        'Accuracy': '70.89%', 'F1 Score': '80.00%', 'Precision': '74.19%', 'Recall': '86.79%', 'ROC-AUC': '64.15%'
    },
    'rf_model.joblib': {
        'name': 'Random Forest Classifier',
        'Accuracy': '68.35%', 'F1 Score': '78.63%', 'Precision': '71.88%', 'Recall': '86.79%', 'ROC-AUC': '64.04%'
    },
    'svm_model.joblib': {
        'name': 'Support Vector Machine (SVM)',
        'Accuracy': '68.35%', 'F1 Score': '79.34%', 'Precision': '70.59%', 'Recall': '90.57%', 'ROC-AUC': '61.39%'
    },
    'knn_model.joblib': {
        'name': 'K-Nearest Neighbors (KNN)',
        'Accuracy': '63.29%', 'F1 Score': '75.21%', 'Precision': '68.75%', 'Recall': '83.02%', 'ROC-AUC': '59.72%'
    }
}

# Load model names
def load_model_names():
    models = []
    if os.path.exists(MODEL_DIR) and os.path.isdir(MODEL_DIR):
        for f_name in os.listdir(MODEL_DIR):
            if f_name.endswith(('.pkl', '.joblib')) and f_name in MODEL_PERFORMANCE_METRICS: # Ensure only models with metrics are loaded
                models.append(f_name)
    else:
        print(f"Warning: Model directory '{MODEL_DIR}' not found or is not a directory. CWD: {os.getcwd()}")
    if not models:
        print(f"Warning: No usable models found in {MODEL_DIR} that match MODEL_PERFORMANCE_METRICS keys.")
    return models

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == USERNAME and password == PASSWORD:
            session['logged_in'] = True
            session['username'] = username
            flash('Logged in successfully!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('index'))
        else:
            flash('Invalid username or password. Please try again.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    model_files = load_model_names()
    # Filter display names to only include loaded models
    active_model_display_names = {mf: MODEL_DISPLAY_NAMES.get(mf, mf.replace('.joblib','').replace('.pkl','').title()) for mf in model_files}
    return render_template('index.html',
                           models=model_files,
                           model_display_names=active_model_display_names,
                           all_features=ALL_FEATURE_NAMES,
                           categorical_features_map=CATEGORICAL_FEATURES_MAP,
                           numerical_features=NUMERICAL_FEATURES,
                           feature_descriptions=FEATURE_DESCRIPTIONS)

@app.route('/about')
@login_required
def about():
    """Render the about page with project information"""
    return render_template('about.html', model_performance_metrics=MODEL_PERFORMANCE_METRICS)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    model_files = load_model_names()
    active_model_display_names = {mf: MODEL_DISPLAY_NAMES.get(mf, mf.replace('.joblib','').replace('.pkl','').title()) for mf in model_files}
    preprocessing_warning = (
        "Note: Numerical features are not scaled and categorical encoding is done manually. "
        "This may affect prediction accuracy. For best results, the original preprocessing pipeline "
        "(including fitted scalers and encoders from your notebook) should be used."
    )

    if request.method == 'POST':
        try:
            selected_model_file = request.form.get('model')
            if not selected_model_file:
                return render_template('index.html', models=model_files, model_display_names=active_model_display_names, all_features=ALL_FEATURE_NAMES, categorical_features_map=CATEGORICAL_FEATURES_MAP, numerical_features=NUMERICAL_FEATURES, feature_descriptions=FEATURE_DESCRIPTIONS, error='Please select a model.', preprocessing_warning=preprocessing_warning)

            model_path = os.path.join(MODEL_DIR, selected_model_file)
            if not os.path.exists(model_path) or selected_model_file not in MODEL_PERFORMANCE_METRICS:
                return render_template('index.html', models=model_files, model_display_names=active_model_display_names, all_features=ALL_FEATURE_NAMES, categorical_features_map=CATEGORICAL_FEATURES_MAP, numerical_features=NUMERICAL_FEATURES, feature_descriptions=FEATURE_DESCRIPTIONS, error=f'Model file {selected_model_file} not found or no metrics available.', preprocessing_warning=preprocessing_warning)

            model = joblib.load(model_path)
            input_features = []
            raw_features_for_display = {}

            for feature_name in ALL_FEATURE_NAMES:
                value = request.form.get(feature_name)
                raw_features_for_display[feature_name] = value
                if value is None or value == '':
                    return render_template('index.html', models=model_files, model_display_names=active_model_display_names, all_features=ALL_FEATURE_NAMES, categorical_features_map=CATEGORICAL_FEATURES_MAP, numerical_features=NUMERICAL_FEATURES, feature_descriptions=FEATURE_DESCRIPTIONS, error=f'Please provide a value for all features. {feature_name.replace("_"," ").title()} is missing.', preprocessing_warning=preprocessing_warning)
                try:
                    if feature_name in CATEGORICAL_FEATURES_MAP:
                        encoded_value = CATEGORICAL_FEATURES_MAP[feature_name].get(value)
                        if encoded_value is None:
                            raise ValueError(f"Invalid value '{value}' for categorical feature '{feature_name.replace('_',' ').title()}'. Check mapping.")
                        input_features.append(encoded_value)
                    else:
                        input_features.append(float(value))
                except ValueError as ve:
                    return render_template('index.html', models=model_files, model_display_names=active_model_display_names, all_features=ALL_FEATURE_NAMES, categorical_features_map=CATEGORICAL_FEATURES_MAP, numerical_features=NUMERICAL_FEATURES, feature_descriptions=FEATURE_DESCRIPTIONS, error=f'Invalid input for {feature_name.replace("_"," ").title()}: {str(ve)}', preprocessing_warning=preprocessing_warning)

            final_features = [np.array(input_features)]
            prediction_result = model.predict(final_features)
            predicted_pass_status = "Pass" if prediction_result[0] == 1 else "Fail"
            if isinstance(prediction_result[0], str):
                 predicted_pass_status = prediction_result[0].title()

            selected_model_display_name = MODEL_DISPLAY_NAMES.get(selected_model_file, selected_model_file)
            model_metrics = MODEL_PERFORMANCE_METRICS.get(selected_model_file)

            return render_template('result.html',
                               prediction=predicted_pass_status,
                               model_name=selected_model_display_name,
                               raw_model_name=selected_model_file,
                               features_display=raw_features_for_display,
                               model_metrics=model_metrics,
                               preprocessing_warning=preprocessing_warning)

        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            return render_template('index.html', models=model_files, model_display_names=active_model_display_names, all_features=ALL_FEATURE_NAMES, categorical_features_map=CATEGORICAL_FEATURES_MAP, numerical_features=NUMERICAL_FEATURES, feature_descriptions=FEATURE_DESCRIPTIONS, error=f'An unexpected error occurred during prediction: {str(e)}', preprocessing_warning=preprocessing_warning)

    return render_template('index.html', models=model_files, model_display_names=active_model_display_names, all_features=ALL_FEATURE_NAMES, categorical_features_map=CATEGORICAL_FEATURES_MAP, numerical_features=NUMERICAL_FEATURES, feature_descriptions=FEATURE_DESCRIPTIONS, preprocessing_warning=preprocessing_warning)

@app.route('/generate_random_input')
@login_required
def generate_random_input():
    random_inputs = {}
    try:
        for feature_name in ALL_FEATURE_NAMES:
            if feature_name in CATEGORICAL_FEATURES_MAP:
                random_inputs[feature_name] = random.choice(list(CATEGORICAL_FEATURES_MAP[feature_name].keys()))
            elif feature_name in FEATURE_RANGES:
                min_val, max_val = FEATURE_RANGES[feature_name]
                if isinstance(min_val, float) or isinstance(max_val, float):
                    random_inputs[feature_name] = round(random.uniform(min_val, max_val), 2)
                else:
                    random_inputs[feature_name] = random.randint(min_val, max_val)
            else: # Should not happen if FEATURE_RANGES is comprehensive
                random_inputs[feature_name] = 0
        return jsonify(random_inputs)
    except Exception as e:
        print(f"Error in /generate_random_input: {e}")
        return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     # Ensure the app is run from the root directory or adjust path to data/models if needed
#     # For example, if running from src/ -> python app.py
#     # The MODEL_DIR is already ../models
#     app.run(debug=True)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import joblib
import numpy as np
import os
import pandas as pd # Added for CSV parsing if needed later, not strictly used for mappings now
import random # Added for random value generation
from functools import wraps # Added for login_required decorator

# Import preprocessing utilities
import sys
# Add project root to sys.path to locate the 'utils' module
# Assuming app.py is in 'src/', so '..' goes to project root
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from utils.preprocessing import (
    apply_label_encoders,
    apply_standard_scaler,
    load_preprocessor,
    load_json_data
)

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
DATASET_PATH = os.path.join(APP_ROOT, '..', 'data', 'student-data.csv') # Path to the dataset
PREPROCESSOR_DIR = os.path.join(APP_ROOT, '..', 'preprocessors')

student_df = None
try:
    if os.path.exists(DATASET_PATH):
        student_df = pd.read_csv(DATASET_PATH)
        print(f"Successfully loaded dataset from {DATASET_PATH}. Shape: {student_df.shape}")
    else:
        print(f"Warning: Dataset file not found at {DATASET_PATH}")
except Exception as e:
    print(f"Error loading dataset: {e}")

# --- Feature Definitions (Used for form generation, descriptions, etc.) ---
ALL_FEATURE_NAMES = [
    'school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
    'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures',
    'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet',
    'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences'
]

# This list MUST match the categorical columns processed with LabelEncoder in the notebook
CATEGORICAL_FEATURE_NAMES_FOR_ENCODING = [
    'school', 'sex', 'address', 'famsize', 'Pstatus',
    'Mjob', 'Fjob', 'reason', 'guardian',
    'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
    'higher', 'internet', 'romantic'
]

# Numerical features will be determined by 'numerical_columns_fitted.json'
# This global will be populated by load_all_preprocessors_on_startup
# Initial placeholder value: features not in categorical list
NUMERICAL_FEATURES = [f for f in ALL_FEATURE_NAMES if f not in CATEGORICAL_FEATURE_NAMES_FOR_ENCODING]

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
    'health': "Current health status (1-very bad to 5-very good)",
    'absences': "Number of school absences (0 to 93)"
}

# For random input generation for categorical features
SIMPLE_CATEGORICAL_VALUES_FOR_RANDOM_GEN = {
    'school': ['GP', 'MS'], 'sex': ['F', 'M'], 'address': ['U', 'R'],
    'famsize': ['LE3', 'GT3'], 'Pstatus': ['T', 'A'],
    'Mjob': ['teacher', 'health', 'services', 'at_home', 'other'],
    'Fjob': ['teacher', 'health', 'services', 'at_home', 'other'],
    'reason': ['home', 'reputation', 'course', 'other'],
    'guardian': ['mother', 'father', 'other'],
    'schoolsup': ['yes', 'no'], 'famsup': ['yes', 'no'], 'paid': ['yes', 'no'],
    'activities': ['yes', 'no'], 'nursery': ['yes', 'no'], 'higher': ['yes', 'no'],
    'internet': ['yes', 'no'], 'romantic': ['yes', 'no']
}

# For random input generation for numerical features (min, max)
FEATURE_RANGES = {
    'age': (15, 22), 'Medu': (0, 4), 'Fedu': (0, 4), 'traveltime': (1, 4),
    'studytime': (1, 4), 'failures': (0, 3), 'famrel': (1, 5),
    'freetime': (1, 5), 'goout': (1, 5), 'Dalc': (1, 5), 'Walc': (1, 5),
    'health': (1, 5), 'absences': (0, 93)
}

# --- Model and Metrics Definitions ---
MODEL_DISPLAY_NAMES = {
    'lr_model.joblib': 'Logistic Regression',
    'rf_model.joblib': 'Random Forest Classifier',
    'knn_model.joblib': 'K-Nearest Neighbors (KNN)',
    'svm_model.joblib': 'Support Vector Machine (SVM)',
    'xgb_model.joblib': 'XGBoost Classifier'
}

MODEL_PERFORMANCE_METRICS = {
    'lr_model.joblib': {'name': 'Logistic Regression', 'Accuracy': '67.09%', 'F1 Score': '77.59%', 'Precision': '71.43%', 'Recall': '84.91%', 'ROC-AUC': '60.09%'},
    'xgb_model.joblib': {'name': 'XGBoost Classifier', 'Accuracy': '70.89%', 'F1 Score': '80.00%', 'Precision': '74.19%', 'Recall': '86.79%', 'ROC-AUC': '64.15%'},
    'rf_model.joblib': {'name': 'Random Forest Classifier', 'Accuracy': '68.35%', 'F1 Score': '78.63%', 'Precision': '71.88%', 'Recall': '86.79%', 'ROC-AUC': '64.04%'},
    'svm_model.joblib': {'name': 'Support Vector Machine (SVM)', 'Accuracy': '68.35%', 'F1 Score': '79.34%', 'Precision': '70.59%', 'Recall': '90.57%', 'ROC-AUC': '61.39%'},
    'knn_model.joblib': {'name': 'K-Nearest Neighbors (KNN)', 'Accuracy': '63.29%', 'F1 Score': '75.21%', 'Precision': '68.75%', 'Recall': '83.02%', 'ROC-AUC': '59.72%'}
}

# --- Globals for Loaded Objects ---
loaded_models = {}
loaded_label_encoders = {} # Dict to store {col_name: LabelEncoder_object}
loaded_standard_scaler = None
fitted_numerical_cols = []    # List of numerical columns the scaler was FITTED on
final_feature_order_from_training = [] # List of all feature columns in order model expects
preprocessors_loaded_successfully = False

# --- Startup Loading Functions ---
def load_all_preprocessors_on_startup():
    global loaded_label_encoders, loaded_standard_scaler, fitted_numerical_cols, final_feature_order_from_training, preprocessors_loaded_successfully, NUMERICAL_FEATURES

    any_critical_error = False
    print("--- Loading Preprocessors ---")

    # Load LabelEncoders
    for col_name in CATEGORICAL_FEATURE_NAMES_FOR_ENCODING:
        encoder_path = os.path.join(PREPROCESSOR_DIR, f"{col_name}_label_encoder.joblib")
        encoder = load_preprocessor(encoder_path) # From utils.preprocessing
        if encoder is None:
            print(f"CRITICAL ERROR: Failed to load LabelEncoder for '{col_name}' from {encoder_path}.")
            any_critical_error = True
        else:
            loaded_label_encoders[col_name] = encoder
            # print(f"Successfully loaded LabelEncoder for '{col_name}'. Classes: {list(encoder.classes_)}")

    # Load StandardScaler
    scaler_path = os.path.join(PREPROCESSOR_DIR, "standard_scaler.joblib")
    loaded_standard_scaler = load_preprocessor(scaler_path)
    if loaded_standard_scaler is None:
        print(f"CRITICAL ERROR: Failed to load StandardScaler from {scaler_path}.")
        any_critical_error = True

    # Load numerical columns list (the ones scaler was fitted on)
    num_cols_path = os.path.join(PREPROCESSOR_DIR, "numerical_columns_fitted.json")
    fitted_numerical_cols_temp = load_json_data(num_cols_path) # From utils.preprocessing
    if fitted_numerical_cols_temp is None:
        print(f"CRITICAL ERROR: Failed to load 'numerical_columns_fitted.json' from {num_cols_path}.")
        any_critical_error = True
    else:
        # Ensure it's a list before extend or assignment
        if isinstance(fitted_numerical_cols_temp, list):
            fitted_numerical_cols.clear() # Clear before extending/assigning
            fitted_numerical_cols.extend(fitted_numerical_cols_temp)
            NUMERICAL_FEATURES = list(fitted_numerical_cols) # Update global for templates
            # print(f"Successfully loaded numerical columns for scaler: {fitted_numerical_cols}")
        else:
            print(f"CRITICAL ERROR: 'numerical_columns_fitted.json' did not load as a list.")
            any_critical_error = True

    # Load final feature order
    order_path = os.path.join(PREPROCESSOR_DIR, "final_feature_order.json")
    final_order_temp = load_json_data(order_path)
    if final_order_temp is None:
        print(f"CRITICAL ERROR: Failed to load 'final_feature_order.json' from {order_path}.")
        any_critical_error = True
    else:
        if isinstance(final_order_temp, list):
            final_feature_order_from_training.clear()
            final_feature_order_from_training.extend(final_order_temp)
            # print(f"Successfully loaded final feature order: {final_feature_order_from_training}")
        else:
            print(f"CRITICAL ERROR: 'final_feature_order.json' did not load as a list.")
            any_critical_error = True

    if not any_critical_error and loaded_standard_scaler and fitted_numerical_cols and final_feature_order_from_training and len(loaded_label_encoders) == len(CATEGORICAL_FEATURE_NAMES_FOR_ENCODING):
        preprocessors_loaded_successfully = True
        print("--- All preprocessors seem to be loaded successfully. ---")
    else:
        preprocessors_loaded_successfully = False
        print("--- CRITICAL: One or more preprocessors FAILED to load. Predictions will be unreliable or fail. ---")

def load_all_models_on_startup():
    global loaded_models
    print("--- Loading Models ---")
    if not (os.path.exists(MODEL_DIR) and os.path.isdir(MODEL_DIR)):
        print(f"CRITICAL ERROR: Model directory '{MODEL_DIR}' not found. CWD: {os.getcwd()}")
        return

    for f_name in os.listdir(MODEL_DIR):
        if f_name.endswith(('.pkl', '.joblib')) and f_name in MODEL_DISPLAY_NAMES: # Use MODEL_DISPLAY_NAMES as source of truth for expected models
            model_path = os.path.join(MODEL_DIR, f_name)
            try:
                loaded_models[f_name] = joblib.load(model_path)
                print(f"Successfully loaded model: {f_name}")
            except Exception as e:
                print(f"Error loading model {f_name} from {model_path}: {e}")
        elif f_name.endswith(('.pkl', '.joblib')):
             print(f"Warning: Model file {f_name} found but not listed in MODEL_DISPLAY_NAMES. Skipping.")

    if not loaded_models:
        print(f"--- CRITICAL WARNING: No models were successfully loaded from {MODEL_DIR}. Predictions will not work. ---")
    else:
        print(f"--- Successfully loaded {len(loaded_models)} models. ---")

# --- Run Startup Loaders ---
load_all_preprocessors_on_startup()
load_all_models_on_startup()

# --- Authentication & Routes ---
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
    dataset_row_count = len(student_df) if student_df is not None else 0
    model_load_error = None
    if not loaded_models:
        model_load_error = f"CRITICAL: No models could be loaded from '{MODEL_DIR}'. Predictions are disabled."

    preprocessor_load_error = None
    if not preprocessors_loaded_successfully:
        preprocessor_load_error = "CRITICAL: One or more essential preprocessors (encoders/scaler/column lists) failed to load. Predictions will be inaccurate or fail. Check server logs."

    # For the template, we need a way to know which features are categorical for dropdowns.
    # We use CATEGORICAL_FEATURE_NAMES_FOR_ENCODING for this.
    # NUMERICAL_FEATURES is now correctly populated from fitted_numerical_cols.
    return render_template('index.html',
                           all_features=ALL_FEATURE_NAMES,
                           # Pass the list of categorical feature names for the template to identify them
                           categorical_feature_names=CATEGORICAL_FEATURE_NAMES_FOR_ENCODING,
                           # numerical_features is now correctly populated from fitted_numerical_cols by load_all_preprocessors_on_startup
                           numerical_features=NUMERICAL_FEATURES,
                           feature_descriptions=FEATURE_DESCRIPTIONS,
                           # For categorical dropdowns, provide the simple string values for options
                           simple_categorical_values=SIMPLE_CATEGORICAL_VALUES_FOR_RANDOM_GEN,
                           dataset_row_count=dataset_row_count,
                           model_load_error=model_load_error,
                           preprocessor_load_error=preprocessor_load_error) # Pass preprocessor error

@app.route('/about')
@login_required
def about():
    """Render the about page with project information"""
    # Load dataset for visualizations
    dataset_path = os.path.join('data', 'student-data.csv')
    dataset_stats = {}

    try:
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)

            # Distribution of parents' education (Medu/Fedu)
            dataset_stats['medu_counts'] = df['Medu'].value_counts().sort_index().to_dict()
            dataset_stats['fedu_counts'] = df['Fedu'].value_counts().sort_index().to_dict()

            # Age distribution
            dataset_stats['age_counts'] = df['age'].value_counts().sort_index().to_dict()

            # Absences distribution (binned)
            absences_bins = [0, 5, 10, 15, 20, 30, 100]
            absences_labels = ['0-5', '6-10', '11-15', '16-20', '21-30', '31+']
            df['absences_binned'] = pd.cut(df['absences'], bins=absences_bins, labels=absences_labels)
            dataset_stats['absences_counts'] = df['absences_binned'].value_counts().to_dict()

            # Alcohol consumption (weekend vs. weekday)
            dataset_stats['walc_counts'] = df['Walc'].value_counts().sort_index().to_dict()
            dataset_stats['dalc_counts'] = df['Dalc'].value_counts().sort_index().to_dict()

            # Pass/Fail distribution
            dataset_stats['pass_fail_counts'] = df['passed'].value_counts().to_dict()

            # Internet access vs. performance
            internet_pass_fail = df.groupby(['internet', 'passed']).size().unstack(fill_value=0)
            dataset_stats['internet_yes_pass'] = internet_pass_fail.loc['yes', 'yes'] if ('yes' in internet_pass_fail.index and 'yes' in internet_pass_fail.columns) else 0
            dataset_stats['internet_yes_fail'] = internet_pass_fail.loc['yes', 'no'] if ('yes' in internet_pass_fail.index and 'no' in internet_pass_fail.columns) else 0
            dataset_stats['internet_no_pass'] = internet_pass_fail.loc['no', 'yes'] if ('no' in internet_pass_fail.index and 'yes' in internet_pass_fail.columns) else 0
            dataset_stats['internet_no_fail'] = internet_pass_fail.loc['no', 'no'] if ('no' in internet_pass_fail.index and 'no' in internet_pass_fail.columns) else 0

    except Exception as e:
        print(f"Error preparing dataset statistics: {e}")
        dataset_stats = {}

    return render_template('about.html', model_performance_metrics=MODEL_PERFORMANCE_METRICS, dataset_stats=dataset_stats)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if not loaded_models:
        flash("No models are loaded. Cannot make predictions.", "danger")
        return redirect(url_for('index'))
    if not preprocessors_loaded_successfully:
        flash("Critical preprocessor components are missing. Predictions cannot be made reliably. Please check server logs.", "danger")
        return redirect(url_for('index'))

    preprocessing_status_message = "Preprocessing will be applied using loaded encoders and scaler."
    if not preprocessors_loaded_successfully: # Should be caught above, but defensive
        preprocessing_status_message = "WARNING: Preprocessing may be incomplete due to loading errors. Results may be inaccurate."

    if request.method == 'POST':
        raw_input_data = {}
        raw_features_for_display = {} # Keep this for result.html
        actual_value_from_dataset = request.form.get('actual_value_from_dataset')

        # 1. Collect and Validate Inputs
        for feature_name in ALL_FEATURE_NAMES:
            value = request.form.get(feature_name)
            raw_features_for_display[feature_name] = value # For display on result page
            if value is None or value == '':
                flash(f'Please provide a value for all features. {feature_name.replace("_"," ").title()} is missing.', 'warning')
                # Simplified redirect for input errors to avoid re-rendering index with complex state here
                return redirect(url_for('index')) # User will lose current inputs, consider JS validation on client-side
            raw_input_data[feature_name] = value

        # 2. Create DataFrame from input
        input_df = pd.DataFrame([raw_input_data])

        # 3. Convert column types for DataFrame before preprocessing
        # Numerical columns (use fitted_numerical_cols loaded from JSON)
        for col in fitted_numerical_cols:
            if col in input_df.columns:
                try:
                    input_df[col] = pd.to_numeric(input_df[col])
                except ValueError:
                    flash(f"Invalid value for numerical feature '{col}': '{input_df[col].iloc[0]}'. Please enter a number.", 'danger')
                    return redirect(url_for('index'))

        # Categorical columns should already be strings/objects, which is fine for LabelEncoder.transform

        # 4. Apply Label Encoders
        # CATEGORICAL_FEATURE_NAMES_FOR_ENCODING defines which columns to encode
        processed_df = apply_label_encoders(
            input_df.copy(), # Pass a copy
            categorical_cols=CATEGORICAL_FEATURE_NAMES_FOR_ENCODING,
            preprocessor_dir=PREPROCESSOR_DIR, # For loading if not pre-loaded
            mode='transform',
            encoders_dict=loaded_label_encoders # Pass pre-loaded encoders
        )
        if processed_df is None:
            flash("Error during categorical data encoding. An invalid value might have been provided for a dropdown. Check inputs.", "danger")
            return redirect(url_for('index'))

        # 5. Apply Standard Scaler
        # fitted_numerical_cols has the list of columns the scaler was FITTED on, in correct order
        processed_df = apply_standard_scaler(
            processed_df, # Use output from label encoding
            numerical_cols=fitted_numerical_cols,
            preprocessor_dir=PREPROCESSOR_DIR, # For loading if not pre-loaded
            mode='transform',
            scaler_object=loaded_standard_scaler # Pass pre-loaded scaler
        )
        if processed_df is None:
            flash("Error during numerical data scaling. Please check inputs.", "danger")
            return redirect(url_for('index'))

        # 6. Reorder Columns to match training order
        try:
            if not final_feature_order_from_training:
                 flash("Internal Server Error: Final feature order for model prediction is not loaded.", "danger")
                 return redirect(url_for('index'))
            processed_df = processed_df[final_feature_order_from_training]
        except KeyError as e:
            print(f"CRITICAL KeyError during column reordering: {e}. This means 'final_feature_order.json' "
                  f"does not match the columns in the processed DataFrame. "
                  f"Processed DF columns: {processed_df.columns.tolist()}. Expected order: {final_feature_order_from_training}")
            flash(f"Internal error: Feature mismatch after preprocessing. Model prediction cannot proceed. (Details: {str(e)})", "danger")
            return redirect(url_for('index'))
        except Exception as e_reorder:
            print(f"CRITICAL Exception during column reordering: {e_reorder}.")
            flash(f"Internal error during feature preparation: {str(e_reorder)}", "danger")
            return redirect(url_for('index'))


        # 7. Convert to NumPy array for prediction
        final_features_array = processed_df.to_numpy() # .reshape(1, -1) might not be needed if df is single row
        if final_features_array.ndim == 1: # Ensure it's 2D for scikit-learn
            final_features_array = final_features_array.reshape(1, -1)

        # Check feature count against one of the models (optional sanity check)
        # Example: first_model_key = next(iter(loaded_models))
        # if loaded_models[first_model_key].n_features_in_ != final_features_array.shape[1]:
        #     flash(f"Feature count mismatch. Model expects {loaded_models[first_model_key].n_features_in_}, got {final_features_array.shape[1]}", "danger")
        #     return redirect(url_for('index'))


        # --- Prediction Loop (largely unchanged) ---
        all_predictions_data = []
        for model_filename, model_object in loaded_models.items():
            display_name = MODEL_DISPLAY_NAMES.get(model_filename, model_filename.replace('.joblib','').replace('.pkl','').title())
            model_metrics = MODEL_PERFORMANCE_METRICS.get(model_filename)
            prediction_label = "Error"
            prob_pass = None
            prob_fail = None

            try:
                if hasattr(model_object, 'predict_proba') and callable(model_object.predict_proba):
                    probabilities = model_object.predict_proba(final_features_array)
                    class_labels = getattr(model_object, 'classes_', [0, 1])
                    idx_pass = 1
                    idx_fail = 0

                    try:
                        # Assuming target 'passed' was encoded as 0 (no/fail) and 1 (yes/pass) in the notebook.
                        # And that model.classes_ reflects this, e.g. [0, 1]
                        # And predict_proba columns correspond to these sorted classes.
                        target_encoder_path = os.path.join(PREPROCESSOR_DIR, "passed_label_encoder.joblib")
                        le_passed = load_preprocessor(target_encoder_path)

                        if le_passed:
                            pass_label_numeric = le_passed.transform(['yes'])[0] # Assuming 'yes' means pass
                            # fail_label_numeric = le_passed.transform(['no'])[0]

                            # Find index of 'pass' in model's classes
                            list_class_labels = list(class_labels) # Convert to list for index()
                            if pass_label_numeric in list_class_labels:
                                idx_pass = list_class_labels.index(pass_label_numeric)
                                idx_fail = 1 - idx_pass # Assuming binary classification
                            else: # Fallback if 'yes' transformed value not in model.classes_
                                print(f"Warning: 'yes' ({pass_label_numeric}) not in model {display_name} classes_ {class_labels}. Using default 0/1 indexing.")
                                # Default idx_pass=1, idx_fail=0 might still be correct if classes are [0,1] and 1 means pass
                        else: # Fallback if target encoder not found
                             print(f"Warning: passed_label_encoder.joblib not found. Using default 0/1 indexing for Pass/Fail probabilities for model {display_name}.")
                             # Default idx_pass=1, idx_fail=0 assumes class 1 is Pass.

                        prob_pass = probabilities[0][idx_pass]
                        prob_fail = probabilities[0][idx_fail]
                        predicted_class_index = np.argmax(probabilities[0])
                        prediction_label = "Pass" if predicted_class_index == idx_pass else "Fail"

                    except ValueError as ve_classes: # e.g. if a class label is not found
                        print(f"Warning: Could not reliably determine Pass/Fail class indices for {display_name} (classes: {class_labels}): {ve_classes}. Falling back to default 0/1 indexing.")
                        prob_pass = probabilities[0][1] # Default: Class 1 probability for "Pass"
                        prob_fail = probabilities[0][0] # Default: Class 0 probability for "Fail"
                        prediction_label = "Pass" if np.argmax(probabilities[0]) == 1 else "Fail"
                else:
                    prediction_result = model_object.predict(final_features_array)
                    # Ensure consistent "Pass"/"Fail" label
                    # Assuming 1 from predict() means "Pass" after target encoding in notebook
                    target_encoder_path = os.path.join(PREPROCESSOR_DIR, "passed_label_encoder.joblib")
                    le_passed = load_preprocessor(target_encoder_path)
                    pass_numeric_val = 1 # Default assumption
                    if le_passed:
                        try:
                            pass_numeric_val = le_passed.transform(['yes'])[0]
                        except Exception:
                            print("Could not transform 'yes' with loaded target encoder, defaulting pass_numeric_val to 1")

                    prediction_label = "Pass" if prediction_result[0] == pass_numeric_val else "Fail"
                    print(f"Note: Model {display_name} does not have predict_proba. Using predict(). Label: {prediction_label} from raw: {prediction_result[0]}")


                all_predictions_data.append({
                    'name': display_name, 'raw_name': model_filename, 'prediction': prediction_label,
                    'prob_pass': prob_pass * 100 if prob_pass is not None else None,
                    'prob_fail': prob_fail * 100 if prob_fail is not None else None,
                    'metrics': model_metrics
                })
            except Exception as model_pred_e:
                print(f"Error predicting with model {model_filename}: {model_pred_e}")
                import traceback
                traceback.print_exc()
                all_predictions_data.append({
                    'name': display_name, 'raw_name': model_filename, 'prediction': 'Error during prediction',
                    'prob_pass': None, 'prob_fail': None, 'metrics': model_metrics, 'error': str(model_pred_e)
                })

        return render_template('result.html',
                               predictions=all_predictions_data,
                               features_display=raw_features_for_display,
                               preprocessing_status=preprocessing_status_message, # Updated from preprocessing_warning
                               actual_value_from_dataset=actual_value_from_dataset)

    # Fallback for GET or other methods to /predict
    return redirect(url_for('index'))

@app.route('/generate_random_input')
@login_required
def generate_random_input():
    random_inputs = {}
    try:
        for feature_name in ALL_FEATURE_NAMES:
            if feature_name in CATEGORICAL_FEATURE_NAMES_FOR_ENCODING: # Use this list
                # Choose from predefined simple values for user-facing categories
                random_inputs[feature_name] = random.choice(SIMPLE_CATEGORICAL_VALUES_FOR_RANDOM_GEN.get(feature_name, ["N/A"]))
            elif feature_name in FEATURE_RANGES: # Numerical features
                min_val, max_val = FEATURE_RANGES[feature_name]
                if isinstance(min_val, float) or isinstance(max_val, float) or feature_name == 'absences': # absences can be int
                     # For simplicity, make all numerical randoms integers if min/max are int, except specific cases
                    if feature_name in ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']:
                         random_inputs[feature_name] = random.randint(int(min_val), int(max_val))
                    else: # If any other numerical, allow float if range suggests
                         random_inputs[feature_name] = round(random.uniform(min_val, max_val), 2)
                else: # Default for int ranges
                    random_inputs[feature_name] = random.randint(min_val, max_val)

            else: # Should not happen if all features are covered
                random_inputs[feature_name] = 0
        return jsonify(random_inputs)
    except Exception as e:
        print(f"Error in /generate_random_input: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_dataset_row/<int:row_index>')
@login_required
def get_dataset_row(row_index):
    if student_df is None:
        return jsonify({"error": "Dataset not loaded."}), 500
    if not 0 <= row_index < len(student_df):
        return jsonify({"error": f"Row index out of bounds. Please select a row between 0 and {len(student_df) - 1}."}), 400

    try:
        row_data = student_df.iloc[row_index].copy() # Use .copy() to avoid SettingWithCopyWarning on potential modifications

        # Convert numpy types to native Python types for JSON serialization
        for col, value in row_data.items():
            if isinstance(value, np.integer):
                row_data[col] = int(value)
            elif isinstance(value, np.floating):
                row_data[col] = float(value)
            elif isinstance(value, np.bool_):
                 row_data[col] = bool(value)

        # The target variable in student-data.csv is 'passed'
        # We need to include this to be sent to the client, so it can be stored and sent back with predict.
        # It's not an input feature itself for the model.
        response_data = row_data.to_dict()

        # Ensure all ALL_FEATURE_NAMES are present, even if some are not in the CSV (though they should be)
        # And add the 'passed' column specifically for the actual value comparison
        final_response = {}
        for feature in ALL_FEATURE_NAMES:
            final_response[feature] = response_data.get(feature)

        if 'passed' in response_data: # This is the target variable
            final_response['actual_value_from_dataset'] = response_data['passed']
        else:
            # This case should ideally not happen if 'passed' is always in your CSV
            final_response['actual_value_from_dataset'] = None
            print(f"Warning: 'passed' column not found in dataset row {row_index}")

        return jsonify(final_response)
    except Exception as e:
        print(f"Error in /get_dataset_row for index {row_index}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Failed to retrieve or process dataset row."}), 500

# if __name__ == '__main__':
#     # Ensure the app is run from the root directory or adjust path to data/models if needed
#     # For example, if running from src/ -> python app.py
#     # The MODEL_DIR is already ../models
#     app.run(debug=True)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

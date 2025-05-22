from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import joblib
import numpy as np
import os
import pandas as pd # Added for CSV parsing if needed later, not strictly used for mappings now
import random # Added for random value generation
from functools import wraps # Added for login_required decorator
from groq import Groq # Added for Groq API
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
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
    'school': "Student's school (binary: \"GP\" or \"MS\")",
    'sex': "Student's sex (binary: \"F\" - female or \"M\" - male)",
    'age': "Student's age (numeric: from 15 to 22)",
    'address': "Student's home address type (binary: \"U\" - urban or \"R\" - rural)",
    'famsize': "Family size (binary: \"LE3\" - less or equal to 3 or \"GT3\" - greater than 3)",
    'Pstatus': "Parent's cohabitation status (binary: \"T\" - living together or \"A\" - apart)",
    'Medu': "Mother's education (numeric: 0 - none,  1 - primary education (4th grade), 2 - 5th to 9th grade, 3 - secondary education or 4 - higher education)",
    'Fedu': "Father's education (numeric: 0 - none,  1 - primary education (4th grade), 2 - 5th to 9th grade, 3 - secondary education or 4 - higher education)",
    'Mjob': "Mother's job (nominal: \"teacher\", \"health\" care related, civil \"services\" (e.g. administrative or police), \"at_home\" or \"other\")",
    'Fjob': "Father's job (nominal: \"teacher\", \"health\" care related, civil \"services\" (e.g. administrative or police), \"at_home\" or \"other\")",
    'reason': "Reason to choose this school (nominal: close to \"home\", school \"reputation\", \"course\" preference or \"other\")",
    'guardian': "Student's guardian (nominal: \"mother\", \"father\" or \"other\")",
    'traveltime': "Home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)",
    'studytime': "Weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)",
    'failures': "Number of past class failures (numeric: n if 1<=n<3, else 4)",
    'schoolsup': "Extra educational support (binary: yes or no)",
    'famsup': "Family educational support (binary: yes or no)",
    'paid': "Extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)",
    'activities': "Extra-curricular activities (binary: yes or no)",
    'nursery': "Attended nursery school (binary: yes or no)",
    'higher': "Wants to take higher education (binary: yes or no)",
    'internet': "Internet access at home (binary: yes or no)",
    'romantic': "With a romantic relationship (binary: yes or no)",
    'famrel': "Quality of family relationships (numeric: from 1 - very bad to 5 - excellent)",
    'freetime': "Free time after school (numeric: from 1 - very low to 5 - very high)",
    'goout': "Going out with friends (numeric: from 1 - very low to 5 - very high)",
    'Dalc': "Workday alcohol consumption (numeric: from 1 - very low to 5 - very high)",
    'Walc': "Weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)",
    'health': "Current health status (numeric: from 1 - very bad to 5 - very good)",
    'absences': "Number of school absences (numeric: from 0 to 93)"
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

# --- Groq API Configuration ---
# IMPORTANT: Set your Groq API key as an environment variable: GROQ_API_KEY
# Example: export GROQ_API_KEY='your_api_key_here'
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY environment variable not set. AI interpretation will be disabled.")
    # You could choose to raise an error or have a fallback.
    # For now, it will just not provide interpretations.

def get_ai_interpretation(input_features: dict, feature_descriptions: dict, all_feature_names: list, prediction_outcome: str = None, conversation_history: list = None, follow_up_message: str = None) -> tuple[str, list]:
    """
    Gets an AI-powered interpretation from Groq API, supporting conversation history and specific behavior for follow-ups.
    For initial calls (no follow_up_message), it expects conversation_history to contain the initial user prompt.
    For follow-ups, it prepends a transient system message to guide concise answers and uses lower temperature.
    """
    if not GROQ_API_KEY:
        error_msg = "AI interpretation is unavailable because the Groq API key is not configured."
        return error_msg, conversation_history or []

    client = Groq(api_key=GROQ_API_KEY)

    # actual_history_to_update will be the conversation history that is maintained and returned.
    actual_history_to_update = conversation_history.copy() if conversation_history else []

    # messages_for_api_call is what we construct to send to the Groq API for the current turn.
    messages_for_api_call = []
    temperature_setting = 0.6 # Default for initial, comprehensive interpretation

    if follow_up_message:
        # This is a follow-up question.
        temperature_setting = 0.3 # Lower temperature for more direct, less verbose follow-ups

        # 1. Prepend the transient system message for THIS API call only.
        messages_for_api_call.append(
            {"role": "system", "content": "You are an AI assistant. The user is asking a follow-up question. Answer it concisely and directly, using the prior conversation for context. Do NOT repeat the full structured analysis or instructions from the initial query. Focus only on the user's latest question to provide a specific answer."}
        )

        # 2. Add the existing actual conversation history for context to the API call.
        messages_for_api_call.extend(actual_history_to_update)

        # 3. Add the user's new follow-up message for the API call.
        user_follow_up_obj = {"role": "user", "content": follow_up_message}
        messages_for_api_call.append(user_follow_up_obj)

        # 4. Also add the user's follow-up to the actual_history_to_update (which will be saved).
        actual_history_to_update.append(user_follow_up_obj)
    else:
        # This is an initial interpretation request.
        # actual_history_to_update (which comes from conversation_history) should already contain the initial user prompt.
        if not actual_history_to_update or actual_history_to_update[-1]["role"] != "user":
            # This defensive check ensures the history passed for an initial call ends with the user's detailed prompt.
            return "Error: Initial prompt not found or history malformed for AI interpretation.", actual_history_to_update
        # For the API call, send the history as is (it contains the initial comprehensive prompt).
        messages_for_api_call = actual_history_to_update

    # Defensive check: ensure there's something to send.
    if not messages_for_api_call or not messages_for_api_call[-1].get("content"):
        return "Error: No message content to send to AI.", actual_history_to_update

    try:
        chat_completion = client.chat.completions.create(
            messages=messages_for_api_call,
            model=GROQ_MODEL,
            temperature=temperature_setting,
            max_tokens=1500, # Max_tokens can be adjusted; 1500 is generous for initial, okay for follow-up.
        )
        ai_response_content = chat_completion.choices[0].message.content

        # Append AI's response to the actual_history_to_update.
        actual_history_to_update.append({"role": "assistant", "content": ai_response_content})
        return ai_response_content, actual_history_to_update
    except Exception as e:
        error_msg = f"An error occurred while communicating with the AI: {str(e)}"
        print(f"Error calling Groq API: {e}")
        # Return the current state of actual_history_to_update, which includes the user's last message if it was a follow-up.
        return error_msg, actual_history_to_update

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
                return redirect(url_for('index'))
            raw_input_data[feature_name] = value

        # 2. Create DataFrame from input
        input_df = pd.DataFrame([raw_input_data])

        # 3. Convert column types for DataFrame before preprocessing
        for col in fitted_numerical_cols:
            if col in input_df.columns:
                try:
                    input_df[col] = pd.to_numeric(input_df[col])
                except ValueError:
                    flash(f"Invalid value for numerical feature '{col}': '{input_df[col].iloc[0]}'. Please enter a number.", 'danger')
                    return redirect(url_for('index'))

        # 4. Apply Label Encoders
        processed_df = apply_label_encoders(
            input_df.copy(),
            categorical_cols=CATEGORICAL_FEATURE_NAMES_FOR_ENCODING,
            preprocessor_dir=PREPROCESSOR_DIR,
            mode='transform',
            encoders_dict=loaded_label_encoders
        )
        if processed_df is None:
            flash("Error during categorical data encoding. An invalid value might have been provided for a dropdown. Check inputs.", "danger")
            return redirect(url_for('index'))

        # 5. Apply Standard Scaler
        processed_df = apply_standard_scaler(
            processed_df,
            numerical_cols=fitted_numerical_cols,
            preprocessor_dir=PREPROCESSOR_DIR,
            mode='transform',
            scaler_object=loaded_standard_scaler
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
        final_features_array = processed_df.to_numpy()
        if final_features_array.ndim == 1:
            final_features_array = final_features_array.reshape(1, -1)

        # --- Prediction Loop ---
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
                        target_encoder_path = os.path.join(PREPROCESSOR_DIR, "passed_label_encoder.joblib")
                        le_passed = load_preprocessor(target_encoder_path)

                        if le_passed:
                            pass_label_numeric = le_passed.transform(['yes'])[0]
                            list_class_labels = list(class_labels)
                            if pass_label_numeric in list_class_labels:
                                idx_pass = list_class_labels.index(pass_label_numeric)
                                idx_fail = 1 - idx_pass
                            else:
                                print(f"Warning: 'yes' ({pass_label_numeric}) not in model {display_name} classes_ {class_labels}. Using default 0/1 indexing.")
                        else:
                             print(f"Warning: passed_label_encoder.joblib not found. Using default 0/1 indexing for Pass/Fail probabilities for model {display_name}.")

                        prob_pass = probabilities[0][idx_pass]
                        prob_fail = probabilities[0][idx_fail]
                        predicted_class_index = np.argmax(probabilities[0])
                        prediction_label = "Pass" if predicted_class_index == idx_pass else "Fail"

                    except ValueError as ve_classes:
                        print(f"Warning: Could not reliably determine Pass/Fail class indices for {display_name} (classes: {class_labels}): {ve_classes}. Falling back to default 0/1 indexing.")
                        prob_pass = probabilities[0][1]
                        prob_fail = probabilities[0][0]
                        prediction_label = "Pass" if np.argmax(probabilities[0]) == 1 else "Fail"
                else:
                    prediction_result = model_object.predict(final_features_array)
                    target_encoder_path = os.path.join(PREPROCESSOR_DIR, "passed_label_encoder.joblib")
                    le_passed = load_preprocessor(target_encoder_path)
                    pass_numeric_val = 1
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

        # Store predictions in session for AI interpretation
        session['last_predictions'] = all_predictions_data
        session['last_features'] = raw_features_for_display

        return render_template('result.html',
                               predictions=all_predictions_data,
                               features_display=raw_features_for_display,
                               preprocessing_status=preprocessing_status_message,
                               actual_value_from_dataset=actual_value_from_dataset)

    return redirect(url_for('index'))

# Define custom_ai_prompt outside of any function so it can be used by multiple functions
def custom_ai_prompt(features, feature_descriptions, all_feature_names, prediction_outcome, all_model_preds):
    features_string = ""
    for feature_name in all_feature_names:
        value = features.get(feature_name, "N/A")
        description = feature_descriptions.get(feature_name, "No description available.")
        features_string += f"{feature_name.replace('_',' ').title()}: {value} (Meaning: {description})\n"
    model_preds_str = "\n".join([f"{k}: {v}" for k, v in all_model_preds.items()])
    prompt = f'''
You are an expert AI assistant. Here are the predictions from several models for a student's performance:
{model_preds_str}

The XGBoost model is the best performing model, so focus your explanation on its prediction: {prediction_outcome}.

IMPORTANT: In your explanation, you MUST explicitly mention and compare the predictions from all models. If all models agree, highlight this consensus. If there are disagreements between models, discuss possible reasons for these differences based on the models' characteristics and the student data.

The student's data is as follows. Use these exact feature names and their meanings for your analysis:
{features_string}

Your response MUST be in a style suitable for a formal report or a word document. DO NOT use any markdown formatting (like asterisks for bold, or # for headings). Instead, use plain text, well-structured paragraphs, and clear topic sentences. You can use line breaks to separate paragraphs or logical sections.

Structure your analysis like this:

Model Predictions Overview:
[First, summarize all model predictions, noting agreements and disagreements. Highlight that XGBoost is the best performing model.]

Overall Assessment of Prediction:
[Start with a clear statement confirming the XGBoost prediction: {prediction_outcome}.]

Key Factors Likely Influencing the Prediction of "{prediction_outcome}":
[Identify 2-4 key features that you believe most significantly contributed to this prediction. For each feature, provide a detailed paragraph explaining HOW and WHY its specific value, in context of its meaning, likely influenced the outcome. Be specific about the values.]

Other Notable Observations:
[Briefly mention any other features that might be relevant, or any seeming contradictions, and how they might be interpreted or why they might be less influential than the key factors.]

Concluding Remarks:
[Provide a concise summary. You can also subtly invite clarification if appropriate, for example: "Further details on any specific aspect can be provided upon request."]

Begin your detailed interpretation now:
'''
    return prompt

@app.route('/get_ai_interpretation', methods=['POST'])
@login_required
def get_ai_interpretation_route():
    """New route to handle AI interpretation requests"""
    if not session.get('last_predictions') or not session.get('last_features'):
        return jsonify({"error": "No prediction data available for interpretation."}), 400

    predictions = session['last_predictions']
    features = session['last_features']

    # Find XGBoost model's prediction
    xgb_pred = None
    for pred in predictions:
        if 'xgb' in pred['raw_name'].lower():
            xgb_pred = pred
            break
    if not xgb_pred:
        # fallback to first model if XGBoost not found
        xgb_pred = predictions[0]

    # Prepare all model predictions as context
    all_model_preds = {p['name']: p['prediction'] for p in predictions}

    # Get the XGBoost prediction outcome
    primary_prediction = xgb_pred['prediction']

    # Compose the initial prompt
    initial_prompt = custom_ai_prompt(features, FEATURE_DESCRIPTIONS, ALL_FEATURE_NAMES, primary_prediction, all_model_preds)
    conversation_history = [{"role": "user", "content": initial_prompt}]

    # Call the AI
    ai_interpretation_text, conversation_history = get_ai_interpretation(
        input_features=features,
        feature_descriptions=FEATURE_DESCRIPTIONS,
        all_feature_names=ALL_FEATURE_NAMES,
        prediction_outcome=primary_prediction,
        conversation_history=conversation_history
    )

    # Store conversation history in session
    session['ai_conversation_history'] = conversation_history

    return jsonify({
        "interpretation": ai_interpretation_text,
        "conversation_history": conversation_history
    })

@app.route('/chat_follow_up', methods=['POST'])
@login_required
def chat_follow_up():
    data = request.get_json()
    follow_up_message = data.get('message')

    # Get conversation history from session
    conversation_history = session.get('ai_conversation_history', [])
    features = session.get('last_features', {})
    predictions = session.get('last_predictions', [])

    # Handle missing conversation history by creating a new conversation
    if not conversation_history and features and predictions:
        # Find XGBoost model's prediction for a new conversation
        xgb_pred = None
        for pred in predictions:
            if 'xgb' in pred['raw_name'].lower():
                xgb_pred = pred
                break
        if not xgb_pred:
            xgb_pred = predictions[0]

        all_model_preds = {p['name']: p['prediction'] for p in predictions}
        primary_prediction = xgb_pred['prediction']

        # Create a new conversation with the same prompt as the initial interpretation
        initial_prompt = custom_ai_prompt(features, FEATURE_DESCRIPTIONS, ALL_FEATURE_NAMES,
                                        primary_prediction, all_model_preds)
        conversation_history = [{"role": "user", "content": initial_prompt}]

        # Add a system message to explain what happened
        conversation_history.append({
            "role": "system",
            "content": "Previous conversation not found. Starting a new conversation."
        })

    if not follow_up_message:
        return jsonify({"error": "Missing message for follow-up."}), 400

    if not conversation_history:
        return jsonify({"error": "No conversation data available. Please refresh the page and try again."}), 400

    if not features:
        return jsonify({"error": "No feature data available for context. Please refresh the page and try again."}), 400

    ai_response_text, updated_history = get_ai_interpretation(
        input_features=features,
        feature_descriptions=FEATURE_DESCRIPTIONS,
        all_feature_names=ALL_FEATURE_NAMES,
        conversation_history=conversation_history,
        follow_up_message=follow_up_message
    )

    # Update conversation history in session
    session['ai_conversation_history'] = updated_history

    return jsonify({
        "reply": ai_response_text,
        "history": updated_history
    })

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

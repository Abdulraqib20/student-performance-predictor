# Student Performance Prediction System: A Flask-Based Web Application

## 1. Abstract

This project details the development of a web-based application designed to predict student academic performance. The system leverages machine learning models to forecast whether a student is likely to pass or fail based on a comprehensive set of demographic, social, and school-related features. The application provides a user-friendly interface for inputting student data, receiving predictions from multiple models, and obtaining an AI-powered interpretation of the results. This report covers the system architecture, core functionalities, technologies employed, and potential areas for future development. The primary goal is to provide an accessible tool for educational stakeholders to identify at-risk students and potentially intervene with targeted support.

## 2. Introduction

Predicting student success is a critical area of research in educational data mining. Early identification of students at risk of academic failure allows for timely interventions, potentially improving student outcomes and retention rates. This project implements a machine learning-based system, packaged as a Flask web application, to make such predictions accessible and interpretable. The system integrates several classification models and utilizes a Large Language Model (LLM) via the Groq API to provide narrative explanations of the predictions, making the insights more actionable for users without a deep statistical background.

## 3. System Architecture

The application follows a modular architecture, primarily consisting of a Flask backend, data storage, machine learning models, preprocessing components, and a web-based frontend.

**3.1. Backend (Flask Application - `src/app.py`)**

The core of the system is a Flask application (`app.py`) that handles:
*   **Routing:** Managing different URL endpoints for login, prediction, data display, and API interactions.
*   **Business Logic:** Implementing the core functionalities such as user authentication, data processing, model inference, and communication with the Groq API.
*   **Session Management:** Maintaining user login state and storing temporary data like recent predictions for AI interpretation.
*   **Serving Frontend:** Rendering HTML templates for the user interface.

**3.2. Data Management (`data/`, `preprocessors/`)**

*   **Dataset (`data/student-data.csv`):** The primary dataset containing student features and historical performance (assumed to have a target variable like 'passed'). This dataset is loaded on application startup for features like fetching random rows.
*   **Preprocessors (`preprocessors/`):** This directory stores serialized preprocessing objects crucial for transforming raw input data into the format expected by the machine learning models. These include:
    *   `LabelEncoder` objects (e.g., `school_label_encoder.joblib`) for converting categorical string features into numerical representations.
    *   A `StandardScaler` object (`standard_scaler.joblib`) for normalizing numerical features.
    *   JSON files (`numerical_columns_fitted.json`, `final_feature_order.json`) that store metadata from the training phase, such as the list of numerical columns the scaler was fitted on and the exact order of features the models expect.
    *   A `LabelEncoder` for the target variable (`passed_label_encoder.joblib`).

**3.3. Machine Learning Models (`models/`)**

The `models/` directory houses pre-trained machine learning models saved using `joblib`. The application supports multiple models, allowing for a comparative view of predictions. Supported models include:
*   Logistic Regression (`lr_model.joblib`)
*   Random Forest Classifier (`rf_model.joblib`)
*   K-Nearest Neighbors (KNN) (`knn_model.joblib`)
*   Support Vector Machine (SVM) (`svm_model.joblib`)
*   XGBoost Classifier (`xgb_model.joblib`)

Model performance metrics (Accuracy, F1-Score, etc.) are also defined within the application to be displayed alongside predictions.

**3.4. Frontend (`src/templates/`, `src/static/`)**

The user interface is built using HTML templates (managed by Jinja2 within Flask) and static assets (CSS, JavaScript).
*   **Templates (`src/templates/`):**
    *   `index.html`: The main page for data input and initiating predictions.
    *   `login.html`: User authentication page.
    *   `result.html`: Displays prediction outcomes from various models, feature importance, and the AI interpretation.
    *   `about.html`: Provides information about the project, dataset visualizations, and model performance metrics.
*   **Static Files (`src/static/`):** Contains CSS for styling and JavaScript for client-side interactions (e.g., fetching random inputs, handling AI chat).

**3.5. Utilities (`utils/preprocessing.py`)**

Helper functions for preprocessing tasks, such as loading preprocessor objects and applying transformations, are encapsulated in `utils/preprocessing.py`. This promotes modularity and code reusability.

## 4. Core Functionalities

**4.1. User Authentication**
A simple username/password based login system restricts access to the application's core features. Session management is handled by Flask to keep users logged in.

**4.2. Data Input and Preprocessing**
Users can input student data through a web form on the main page (`index.html`). The application supports:
*   **Manual Input:** Users fill in values for all required features.
*   **Random Input Generation:** A feature to populate the form with random valid data, useful for testing or exploration.
*   **Fetching from Dataset:** Users can load a specific row from the underlying dataset into the form.

Upon submission, the raw input data undergoes a series of preprocessing steps:
1.  **Type Conversion:** Numerical features are converted to their appropriate numeric types.
2.  **Label Encoding:** Categorical features are transformed into numerical representations using the pre-loaded `LabelEncoder` objects.
3.  **Standard Scaling:** Numerical features are scaled using the pre-loaded `StandardScaler`.
4.  **Column Reordering:** Features are arranged into the specific order that the models were trained on, as defined in `final_feature_order.json`.

**4.3. Model Prediction**
The preprocessed data (a NumPy array) is then fed to each of the loaded machine learning models. The application:
*   Loads all models from the `models/` directory at startup.
*   Iterates through each model to get a prediction (Pass/Fail).
*   For models that support it, it also calculates prediction probabilities (probability of Pass and Fail).
*   Handles potential errors during prediction for each model gracefully.

**4.4. AI-Powered Interpretation (Groq API Integration)**
A key feature is the integration with the Groq API to provide natural language interpretations of the prediction results.
*   The system constructs a detailed prompt for the LLM (e.g., `meta-llama/llama-4-scout-17b-16e-instruct`), including:
    *   Predictions from all models.
    *   The specific prediction of the best-performing model (XGBoost by default).
    *   The input student features and their descriptions.
    *   A structured format for the desired output.
*   The `get_ai_interpretation` function in `app.py` handles the API call.
*   **Conversational Follow-up:** The system supports follow-up questions to the AI. Conversation history is maintained in the session to provide context for subsequent queries, allowing users to ask clarifying questions about the initial interpretation. A transient system message is used for follow-ups to guide the AI towards concise answers.

**4.5. Display of Results (`result.html`)**
The prediction results are presented on the `result.html` page, showing:
*   The input features.
*   A table of predictions from all models, including probabilities (if available) and key performance metrics for each model.
*   The AI-generated interpretation.
*   An interface for asking follow-up questions to the AI.
*   The actual outcome if the data was loaded from the dataset (for comparison).

**4.6. About Page (`about.html`)**
This page provides:
*   General information about the project.
*   Visualizations and statistics derived from the `student-data.csv` dataset, offering insights into distributions of features like parental education, age, absences, alcohol consumption, and internet access versus performance.
*   A summary table of the performance metrics for all integrated machine learning models.

**4.7. Random Input Generation and Dataset Row Fetching**
Endpoints `/generate_random_input` and `/get_dataset_row/<int:row_index>` provide utility:
*   `/generate_random_input`: Returns a JSON object with randomly generated valid values for all input features, respecting defined ranges and categorical options.
*   `/get_dataset_row/<int:row_index>`: Fetches a specific row from the `student-df` (loaded from `student-data.csv`) and returns its data as JSON, including the actual 'passed' value for comparison. This data can then populate the input form.

## 5. Technologies Used

*   **Backend:**
    *   Python 3.x
    *   Flask: Micro web framework for building the application.
    *   Joblib: For saving and loading machine learning models and preprocessors.
    *   NumPy: For numerical operations, especially handling the feature array for models.
    *   Pandas: For data manipulation, loading the dataset, and creating DataFrames from input.
    *   Groq SDK (`groq`): For interacting with the Groq API to get AI interpretations.
    *   python-dotenv: For managing environment variables (like API keys).
*   **Frontend:**
    *   HTML5
    *   CSS3 (Potentially with a framework like Bootstrap, or custom styles)
    *   JavaScript: For client-side interactivity (e.g., AJAX calls for AI interpretation, random data generation).
*   **Machine Learning Workflow (Implicit, from model/preprocessor generation):**
    *   Scikit-learn: Likely used for training the models (Logistic Regression, RF, KNN, SVM) and for preprocessing (LabelEncoder, StandardScaler).
    *   XGBoost: For the XGBoost model.
*   **Development & Deployment:**
    *   Git: For version control.
    *   Procfile & `requirements.txt`: For deployment (e.g., on platforms like Heroku or Vercel).
    *   Vercel (`vercel.json`): Configuration for deploying on Vercel.

## 6. Setup and Running the Application

1.  **Prerequisites:**
    *   Python 3.x installed.
    *   Git installed.
2.  **Clone Repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
3.  **Create Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Environment Variables:**
    *   Create a `.env` file in the project root.
    *   Add your Groq API key: `GROQ_API_KEY='your_groq_api_key_here'`
6.  **Ensure Data, Models, and Preprocessors are in Place:**
    *   The `data/student-data.csv` file should exist.
    *   The `models/` directory should contain the `.joblib` model files.
    *   The `preprocessors/` directory should contain all necessary `.joblib` and `.json` preprocessor files.
7.  **Run the Application:**
    *   Navigate to the `src/` directory: `cd src`
    *   Run the Flask app: `python app.py`
    *   The application will typically be available at `http://127.0.0.1:5000/` or `http://0.0.0.0:<port>/`.

## 7. Application Flow Example

1.  User navigates to the application URL and is redirected to the login page.
2.  User logs in with valid credentials (USERNAME, PASSWORD defined in `app.py`).
3.  User is redirected to the main page (`index.html`).
4.  User either manually inputs student data, generates random data, or loads a row from the dataset.
5.  User submits the form.
6.  Backend (`/predict` route):
    *   Collects and validates input.
    *   Applies label encoding to categorical features.
    *   Applies standard scaling to numerical features.
    *   Reorders features to match model training order.
    *   Feeds the processed features to each loaded ML model.
    *   Collects predictions and probabilities.
7.  The results, including all model predictions and input features, are displayed on `result.html`.
8.  User clicks "Get AI Interpretation".
9.  Frontend sends an AJAX request to `/get_ai_interpretation`.
10. Backend:
    *   Retrieves last predictions and features from the session.
    *   Constructs a detailed prompt for the Groq API.
    *   Calls the Groq API.
    *   Receives the AI-generated text.
    *   Stores the conversation history in the session.
11. The AI interpretation is displayed on the `result.html` page.
12. User can ask follow-up questions via the chat interface. Frontend sends requests to `/chat_follow_up`.
13. Backend uses existing conversation history and the new user message to get a contextualized response from Groq.

## 8. Conclusion

This project successfully implements a student performance prediction system with an intuitive web interface and an innovative AI-powered interpretation feature. By combining multiple machine learning models and leveraging the capabilities of LLMs, the application provides comprehensive insights into factors influencing student success. The modular architecture allows for easy maintenance and future expansion.

## 9. Future Work and Potential Enhancements

*   **Advanced User Management:** Implement role-based access and allow users to register and manage their accounts.
*   **Model Retraining Interface:** Allow administrators to retrain models with new data through the UI.
*   **Expanded Dataset and Features:** Incorporate a wider range of features or datasets from different educational contexts.
*   **Personalized Feedback:** Tailor AI interpretations more specifically based on individual student history if available.
*   **Batch Predictions:** Allow users to upload a CSV file for predicting multiple students at once.
*   **Enhanced Visualizations:** Integrate more sophisticated interactive visualizations for data exploration and model performance on the "About" page or results page.
*   **A/B Testing Models:** Implement a framework for A/B testing new models against existing ones.
*   **Security Enhancements:** Implement more robust security measures beyond basic authentication, especially if handling sensitive student data.
*   **CI/CD Pipeline:** Set up a Continuous Integration/Continuous Deployment pipeline for automated testing and deployment.
*   **More Sophisticated AI Prompting:** Experiment with more advanced prompting techniques for the Groq API to elicit even more nuanced or targeted explanations.

---
*This report structure is intended as a comprehensive guide for understanding the project. Specific details may vary based on the exact implementation and evolutionary state of the codebase.*

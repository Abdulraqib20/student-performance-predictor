import os
import joblib
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- Save/Load Utilities ---
def save_preprocessor(preprocessor, filepath):
    """Saves a preprocessor (encoder/scaler) to a file."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(preprocessor, filepath)
        print(f"Successfully saved preprocessor to {filepath}")
    except Exception as e:
        print(f"Error saving preprocessor to {filepath}: {e}")

def load_preprocessor(filepath):
    """Loads a preprocessor (encoder/scaler) from a file."""
    try:
        if not os.path.exists(filepath):
            print(f"Error: Preprocessor file not found at {filepath}")
            return None
        preprocessor = joblib.load(filepath)
        print(f"Successfully loaded preprocessor from {filepath}")
        return preprocessor
    except Exception as e:
        print(f"Error loading preprocessor from {filepath}: {e}")
        return None

def save_json_data(data, filepath):
    """Saves Python data (list/dict) to a JSON file."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Successfully saved JSON data to {filepath}")
    except Exception as e:
        print(f"Error saving JSON data to {filepath}: {e}")

def load_json_data(filepath):
    """Loads Python data (list/dict) from a JSON file."""
    try:
        if not os.path.exists(filepath):
            print(f"Error: JSON file not found at {filepath}")
            return None
        with open(filepath, 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded JSON data from {filepath}")
        return data
    except Exception as e:
        print(f"Error loading JSON data from {filepath}: {e}")
        return None

# --- Core Preprocessing Functions ---

def apply_label_encoders(df, categorical_cols, preprocessor_dir, mode='transform', encoders_dict=None):
    """
    Applies LabelEncoders to specified categorical columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        categorical_cols (list): List of categorical column names to encode.
        preprocessor_dir (str): Directory to save/load encoders.
        mode (str): 'fit_transform_save' to fit, transform and save new encoders.
                    'transform' to load and apply existing encoders.
        encoders_dict (dict, optional): Pre-loaded dictionary of {col_name: encoder_object}.
                                        Used if mode is 'transform' and encoders are already loaded.

    Returns:
        pd.DataFrame or None: DataFrame with encoded columns, or None on critical failure.
        dict or None: Dictionary of fitted LabelEncoders if mode is 'fit_transform_save'. None otherwise.
    """
    df_processed = df.copy()

    # Ensure all categorical_cols exist in the DataFrame for 'fit_transform_save'
    if mode == 'fit_transform_save':
        missing_cols_fit = [col for col in categorical_cols if col not in df_processed.columns]
        if missing_cols_fit:
            print(f"Error (fit_transform_save): The following categorical columns are not in the DataFrame: {missing_cols_fit}. Cannot proceed.")
            return None, None

    fitted_encoders = {}

    if mode == 'fit_transform_save':
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            fitted_encoders[col] = le
            encoder_path = os.path.join(preprocessor_dir, f"{col}_label_encoder.joblib")
            save_preprocessor(le, encoder_path)
        return df_processed, fitted_encoders

    elif mode == 'transform':
        current_encoders = encoders_dict or {}
        if not encoders_dict: # Load them if not provided
            for col in categorical_cols:
                # Only try to load/use encoders for columns actually present in the input df for transform mode
                if col not in df_processed.columns:
                    # print(f"Info (transform): Column '{col}' not in input DataFrame. Skipping encoder load/transform for it.")
                    continue
                encoder_path = os.path.join(preprocessor_dir, f"{col}_label_encoder.joblib")
                encoder = load_preprocessor(encoder_path)
                if encoder is None:
                    print(f"Critical Error (transform): Failed to load encoder for column '{col}'. Cannot proceed.")
                    return None
                current_encoders[col] = encoder

        for col in categorical_cols:
            if col not in df_processed.columns:
                # This column was not in the input df, so skip.
                continue
            if col not in current_encoders:
                print(f"Critical Error (transform): Encoder for column '{col}' is required but not available in provided/loaded encoders. Cannot transform.")
                return None

            le = current_encoders[col]
            try:
                # Transform known values. Handle unseen values by checking against classes.
                # For a single row DataFrame typical in Flask app:
                value_to_transform = df_processed[col].iloc[0]
                if value_to_transform not in le.classes_:
                    print(f"Error (transform): Value '{value_to_transform}' in column '{col}' was not seen during training. Cannot transform.")
                    # You might want to flash a user-friendly message here or map to a special value if your model handles it.
                    return None # Indicate failure for this row
                df_processed[col] = le.transform(df_processed[col])
            except Exception as e: # Broader exception for other transform issues
                print(f"Error transforming column '{col}': {e}")
                return None
        return df_processed

    else:
        raise ValueError("Invalid mode for apply_label_encoders. Choose 'fit_transform_save' or 'transform'.")


def apply_standard_scaler(df, numerical_cols, preprocessor_dir, mode='transform', scaler_object=None):
    """
    Applies StandardScaler to specified numerical columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        numerical_cols (list): List of numerical column names that the scaler was/should be fitted on.
        preprocessor_dir (str): Directory to save/load the scaler.
        mode (str): 'fit_transform_save' to fit, transform and save a new scaler.
                    'transform' to load and apply an existing scaler.
        scaler_object (StandardScaler, optional): Pre-loaded StandardScaler object.
                                               Used if mode is 'transform' and scaler is already loaded.
    Returns:
        pd.DataFrame or None: DataFrame with scaled columns, or None on critical failure.
        StandardScaler or None: Fitted StandardScaler if mode is 'fit_transform_save'. None otherwise.
    """
    df_processed = df.copy()

    if not numerical_cols:
        print("Warning: No numerical columns specified for scaling.")
        if mode == 'fit_transform_save':
            return df_processed, None
        return df_processed

    # Identify which of the expected numerical_cols are actually in the current df
    actual_cols_in_df = [col for col in numerical_cols if col in df_processed.columns]

    if not actual_cols_in_df:
        print(f"Warning: None of the specified numerical columns {numerical_cols} found in input DataFrame. Scaling skipped.")
        if mode == 'fit_transform_save': # Cannot fit if no columns
            print("Error (fit_transform_save): Cannot fit scaler with no matching columns in DataFrame.")
            return None, None
        return df_processed # Return as is for transform if no relevant columns

    # For 'fit_transform_save', all listed numerical_cols must be present
    if mode == 'fit_transform_save':
        missing_cols_for_fit = [col for col in numerical_cols if col not in df_processed.columns]
        if missing_cols_for_fit:
            print(f"Error (fit_transform_save): The following numerical columns expected for fitting are not in the DataFrame: {missing_cols_for_fit}. Cannot proceed.")
            return None, None

        # Use the original numerical_cols list for fitting, as they all must be there
        cols_to_fit_on = numerical_cols
        scaler = StandardScaler()
        df_processed[cols_to_fit_on] = scaler.fit_transform(df_processed[cols_to_fit_on])

        scaler_path = os.path.join(preprocessor_dir, "standard_scaler.joblib")
        save_preprocessor(scaler, scaler_path)

        num_cols_path = os.path.join(preprocessor_dir, "numerical_columns_fitted.json")
        save_json_data(cols_to_fit_on, num_cols_path)
        return df_processed, scaler

    elif mode == 'transform':
        current_scaler = scaler_object
        if not current_scaler:
            scaler_path = os.path.join(preprocessor_dir, "standard_scaler.joblib")
            current_scaler = load_preprocessor(scaler_path)

        if current_scaler is None:
            print("Critical Error (transform): Failed to load StandardScaler. Cannot proceed.")
            return None

        # For transform, we operate on the intersection: `actual_cols_in_df`
        # The loaded scaler knows how many features it expects.
        # The critical part is that `df_processed[actual_cols_in_df]` must match the columns
        # the scaler was fitted on *in number and order* if `actual_cols_in_df` is what's passed.
        # It's safer to ensure the input df to transform has exactly the columns the scaler was fitted on.
        # This will be handled by app.py ensuring 'numerical_cols' matches the list from 'numerical_columns_fitted.json'

        if not actual_cols_in_df: # Should have been caught earlier, but defensive check
             print("Info (transform): No relevant numerical columns in DataFrame to scale. Returning as is.")
             return df_processed

        try:
            # Ensure the DataFrame subset being transformed has columns in the order the scaler expects.
            # The `numerical_cols` argument for this function in 'transform' mode *must* be the
            # list of columns the scaler was originally fitted on, in the correct order.
            df_processed[numerical_cols] = current_scaler.transform(df_processed[numerical_cols])
        except ValueError as e:
            print(f"Error transforming numerical columns: {e}. Ensure the columns and their order match the scaler's training.")
            return None
        return df_processed

    else:
        raise ValueError("Invalid mode for apply_standard_scaler. Choose 'fit_transform_save' or 'transform'.")

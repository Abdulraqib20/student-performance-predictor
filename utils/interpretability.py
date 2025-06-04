"""
Advanced ML Model Interpretability Module
Implements state-of-the-art explainability techniques including SHAP, LIME, PDPs, and feature importance analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import shap
import lime
import lime.lime_tabular
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.preprocessing import LabelEncoder
import joblib
import io
import base64
import json
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend for server environments
plt.switch_backend('Agg')

class ModelInterpreter:
    """
    Comprehensive model interpretability class that provides multiple explainability methods.
    """

    def __init__(self, models_dict: Dict, training_data: pd.DataFrame, feature_names: List[str],
                 feature_descriptions: Dict[str, str], categorical_features: List[str],
                 preprocessors: Dict = None):
        """
        Initialize the interpreter with models and data.

        Args:
            models_dict: Dictionary of trained models {model_name: model_object}
            training_data: Training dataset for baseline calculations
            feature_names: List of feature names in order
            feature_descriptions: Dictionary mapping feature names to descriptions
            categorical_features: List of categorical feature names
            preprocessors: Dictionary of preprocessing objects
        """
        self.models = models_dict
        self.training_data = training_data
        self.feature_names = feature_names
        self.feature_descriptions = feature_descriptions
        self.categorical_features = categorical_features
        self.preprocessors = preprocessors or {}

        # Initialize SHAP explainers for each model
        self.shap_explainers = {}
        self._initialize_shap_explainers()

        # Initialize LIME explainer
        self._initialize_lime_explainer()

    def _initialize_shap_explainers(self):
        """Initialize SHAP explainers for each model type."""
        for model_name, model in self.models.items():
            try:
                model_type = type(model).__name__.lower()

                if 'tree' in model_type or 'forest' in model_type or 'xgb' in model_type or 'gradient' in model_type:
                    # Tree-based models
                    self.shap_explainers[model_name] = shap.TreeExplainer(model)
                elif 'linear' in model_type or 'logistic' in model_type:
                    # Linear models
                    self.shap_explainers[model_name] = shap.LinearExplainer(model, self.training_data)
                else:
                    # Default to Kernel explainer for other models
                    # Use a sample for faster computation
                    background_sample = shap.sample(self.training_data, min(100, len(self.training_data)))
                    self.shap_explainers[model_name] = shap.KernelExplainer(model.predict_proba, background_sample)

                print(f"SHAP explainer initialized for {model_name}")
            except Exception as e:
                print(f"Warning: Could not initialize SHAP explainer for {model_name}: {e}")

    def _initialize_lime_explainer(self):
        """Initialize LIME explainer."""
        try:
            categorical_features_indices = [i for i, name in enumerate(self.feature_names)
                                          if name in self.categorical_features]

            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                self.training_data.values,
                feature_names=self.feature_names,
                categorical_features=categorical_features_indices,
                mode='classification',
                verbose=False
            )
            print("LIME explainer initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize LIME explainer: {e}")
            self.lime_explainer = None

    def calculate_shap_values(self, instance_data: np.ndarray, model_name: str) -> Dict[str, Any]:
        """
        Calculate SHAP values for a single instance.

        Args:
            instance_data: Single instance data as numpy array
            model_name: Name of the model to explain

        Returns:
            Dictionary containing SHAP values and related information
        """
        if model_name not in self.shap_explainers:
            return {"error": f"SHAP explainer not available for {model_name}"}

        try:
            explainer = self.shap_explainers[model_name]
            shap_values = explainer.shap_values(instance_data.reshape(1, -1))

            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # For binary classification, take the positive class
                if len(shap_values) == 2:
                    shap_values = shap_values[1]  # Positive class
                else:
                    shap_values = shap_values[0]  # Default to first

            # Ensure it's a 1D array for single instance
            if shap_values.ndim > 1:
                shap_values = shap_values[0]  # Get values for the single instance

            # Create feature importance ranking
            feature_importance = list(zip(self.feature_names, shap_values))
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

            # Get base value safely
            base_value = 0
            if hasattr(explainer, 'expected_value'):
                expected_val = explainer.expected_value
                if isinstance(expected_val, np.ndarray):
                    base_value = expected_val[0] if len(expected_val) > 0 else 0
                elif isinstance(expected_val, (list, tuple)):
                    base_value = expected_val[1] if len(expected_val) > 1 else (expected_val[0] if len(expected_val) > 0 else 0)
                else:
                    base_value = float(expected_val) if expected_val is not None else 0

            return {
                "shap_values": shap_values.tolist(),
                "feature_importance": feature_importance,
                "base_value": base_value,
                "feature_names": self.feature_names
            }
        except Exception as e:
            return {"error": f"Error calculating SHAP values: {str(e)}"}

    def calculate_lime_explanation(self, instance_data: np.ndarray, model, model_name: str) -> Dict[str, Any]:
        """
        Calculate LIME explanation for a single instance.

        Args:
            instance_data: Single instance data as numpy array
            model: The model object to explain
            model_name: Name of the model

        Returns:
            Dictionary containing LIME explanation
        """
        if self.lime_explainer is None:
            return {"error": "LIME explainer not available"}

        try:
            explanation = self.lime_explainer.explain_instance(
                instance_data,
                model.predict_proba,
                num_features=len(self.feature_names)
            )

            # Extract explanation data
            explanation_list = explanation.as_list()

            return {
                "explanations": explanation_list,
                "score": explanation.score if hasattr(explanation, 'score') else 0,
                "intercept": explanation.intercept[1] if hasattr(explanation, 'intercept') and len(explanation.intercept) > 1 else 0
            }
        except Exception as e:
            return {"error": f"Error calculating LIME explanation: {str(e)}"}

    def calculate_permutation_importance(self, X_test: np.ndarray, y_test: np.ndarray,
                                       model, scoring: str = 'accuracy') -> Dict[str, Any]:
        """
        Calculate permutation feature importance.

        Args:
            X_test: Test features
            y_test: Test labels
            model: Model to analyze
            scoring: Scoring metric

        Returns:
            Dictionary containing permutation importance results
        """
        try:
            perm_importance = permutation_importance(
                model, X_test, y_test,
                n_repeats=10, random_state=42, scoring=scoring
            )

            feature_importance = list(zip(
                self.feature_names,
                perm_importance.importances_mean,
                perm_importance.importances_std
            ))
            feature_importance.sort(key=lambda x: x[1], reverse=True)

            return {
                "feature_importance": feature_importance,
                "importances_mean": perm_importance.importances_mean.tolist(),
                "importances_std": perm_importance.importances_std.tolist()
            }
        except Exception as e:
            return {"error": f"Error calculating permutation importance: {str(e)}"}

    def generate_feature_importance_plot(self, shap_values: List[float], top_n: int = 10) -> str:
        """
        Generate feature importance plot as base64 encoded image.

        Args:
            shap_values: SHAP values for features
            top_n: Number of top features to display

        Returns:
            Base64 encoded image string
        """
        try:
            # Create feature importance data
            feature_importance = list(zip(self.feature_names, shap_values))
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            top_features = feature_importance[:top_n]

            # Create plot
            plt.figure(figsize=(10, 6))
            features, values = zip(*top_features)
            colors = ['red' if v < 0 else 'blue' for v in values]

            plt.barh(range(len(features)), values, color=colors, alpha=0.7)
            plt.yticks(range(len(features)), [f.replace('_', ' ').title() for f in features])
            plt.xlabel('SHAP Value (Impact on Prediction)')
            plt.title('Feature Importance (SHAP Values)')
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            plt.tight_layout()

            # Convert to base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()

            return img_base64
        except Exception as e:
            print(f"Error generating feature importance plot: {e}")
            return ""

    def generate_interactive_shap_plot(self, shap_values: List[float], instance_data: List[float]) -> str:
        """
        Generate interactive SHAP waterfall plot using Plotly.

        Args:
            shap_values: SHAP values for features
            instance_data: Feature values for the instance

        Returns:
            JSON string of Plotly figure
        """
        try:
            # Prepare data for waterfall plot
            feature_importance = list(zip(self.feature_names, shap_values, instance_data))
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

            # Take top 10 features for clarity
            top_features = feature_importance[:10]

            features = []
            values = []
            feature_values = []

            for feature, shap_val, feat_val in top_features:
                features.append(feature.replace('_', ' ').title())
                values.append(shap_val)
                feature_values.append(feat_val)

            # Create waterfall plot
            fig = go.Figure(go.Waterfall(
                name="SHAP Values",
                orientation="h",
                y=features,
                x=values,
                textposition="outside",
                text=[f"{v:.3f}" for v in values],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
            ))

            fig.update_layout(
                title="SHAP Values - Feature Contribution to Prediction",
                xaxis_title="SHAP Value (Impact on Prediction)",
                yaxis_title="Features",
                height=400,
                margin=dict(l=150, r=50, t=50, b=50)
            )

            return fig.to_json()
        except Exception as e:
            print(f"Error generating interactive SHAP plot: {e}")
            return "{}"

    def calculate_partial_dependence(self, model, feature_name: str, X_data: np.ndarray) -> Dict[str, Any]:
        """
        Calculate partial dependence for a specific feature.

        Args:
            model: Model to analyze
            feature_name: Name of the feature
            X_data: Input data

        Returns:
            Dictionary containing partial dependence results
        """
        try:
            feature_idx = self.feature_names.index(feature_name)

            pd_result = partial_dependence(
                model, X_data, [feature_idx],
                kind='average', grid_resolution=20
            )

            return {
                "values": pd_result['values'][0].tolist(),
                "grid_values": pd_result['grid_values'][0].tolist(),
                "feature_name": feature_name
            }
        except Exception as e:
            return {"error": f"Error calculating partial dependence: {str(e)}"}

    def generate_comprehensive_report(self, instance_data: np.ndarray, instance_features: Dict[str, Any],
                                    model_predictions: Dict[str, Any], primary_model_name: str) -> Dict[str, Any]:
        """
        Generate a comprehensive interpretability report for a single prediction.

        Args:
            instance_data: Processed feature data
            instance_features: Raw feature values
            model_predictions: Dictionary of model predictions
            primary_model_name: Name of the primary model to focus on

        Returns:
            Comprehensive interpretability report
        """
        report = {
            "instance_features": instance_features,
            "model_predictions": model_predictions,
            "primary_model": primary_model_name,
            "interpretability_analysis": {}
        }

        # Get primary model
        if primary_model_name in self.models:
            primary_model = self.models[primary_model_name]

            # Calculate SHAP values
            shap_results = self.calculate_shap_values(instance_data, primary_model_name)
            if "error" not in shap_results:
                report["interpretability_analysis"]["shap"] = shap_results

                # Generate SHAP plots
                shap_plot = self.generate_feature_importance_plot(shap_results["shap_values"])
                if shap_plot:
                    report["interpretability_analysis"]["shap_plot"] = shap_plot

                interactive_plot = self.generate_interactive_shap_plot(
                    shap_results["shap_values"], instance_data.tolist()
                )
                if interactive_plot != "{}":
                    report["interpretability_analysis"]["interactive_plot"] = interactive_plot

            # Calculate LIME explanation
            lime_results = self.calculate_lime_explanation(instance_data, primary_model, primary_model_name)
            if "error" not in lime_results:
                report["interpretability_analysis"]["lime"] = lime_results

        # Add feature descriptions and insights
        report["interpretability_analysis"]["feature_insights"] = self._generate_feature_insights(
            instance_features, shap_results.get("feature_importance", [])
        )

        return report

    def _generate_feature_insights(self, instance_features: Dict[str, Any],
                                 feature_importance: List[Tuple[str, float]]) -> Dict[str, str]:
        """
        Generate human-readable insights about feature contributions.

        Args:
            instance_features: Raw feature values
            feature_importance: List of (feature_name, shap_value) tuples

        Returns:
            Dictionary of feature insights
        """
        insights = {}

        for feature_name, shap_value in feature_importance[:5]:  # Top 5 features
            feature_value = instance_features.get(feature_name, "Unknown")
            description = self.feature_descriptions.get(feature_name, "No description available")

            direction = "increases" if shap_value > 0 else "decreases"
            magnitude = "strongly" if abs(shap_value) > 0.1 else "moderately" if abs(shap_value) > 0.05 else "slightly"

            insight = f"The feature '{feature_name.replace('_', ' ').title()}' with value '{feature_value}' {magnitude} {direction} the likelihood of passing. {description}"
            insights[feature_name] = insight

        return insights

    def create_enhanced_ai_prompt(self, interpretability_report: Dict[str, Any]) -> str:
        """
        Create an enhanced AI prompt with quantitative interpretability data.

        Args:
            interpretability_report: Comprehensive interpretability report

        Returns:
            Enhanced prompt string with interpretability insights
        """
        model_preds = interpretability_report["model_predictions"]
        features = interpretability_report["instance_features"]
        primary_model = interpretability_report["primary_model"]

        # Extract SHAP insights
        shap_insights = ""
        if "shap" in interpretability_report["interpretability_analysis"]:
            shap_data = interpretability_report["interpretability_analysis"]["shap"]
            top_features = shap_data["feature_importance"][:5]

            shap_insights = "\n\nQUANTITATIVE FEATURE IMPORTANCE (SHAP Analysis):\n"
            for i, (feature, shap_value) in enumerate(top_features, 1):
                direction = "POSITIVE" if shap_value > 0 else "NEGATIVE"
                shap_insights += f"{i}. {feature.replace('_', ' ').title()}: {shap_value:.4f} ({direction} impact)\n"

        # Extract LIME insights
        lime_insights = ""
        if "lime" in interpretability_report["interpretability_analysis"]:
            lime_data = interpretability_report["interpretability_analysis"]["lime"]
            lime_insights = "\n\nLOCAL EXPLANATION (LIME Analysis):\n"
            for feature, contribution in lime_data["explanations"][:5]:
                lime_insights += f"- {feature}: {contribution:.4f}\n"

        # Create feature string with enhanced context
        features_string = ""
        for feature_name, value in features.items():
            description = self.feature_descriptions.get(feature_name, "No description available.")
            features_string += f"{feature_name.replace('_', ' ').title()}: {value} (Meaning: {description})\n"

        model_preds_str = "\n".join([f"{k}: {v}" for k, v in model_preds.items()])

        prompt = f'''
You are an expert AI assistant specializing in educational data analysis and machine learning interpretability. You have access to advanced quantitative analysis including SHAP values and LIME explanations.

PREDICTION RESULTS:
{model_preds_str}

The {primary_model} is the best performing model, so focus your explanation on its prediction.

ADVANCED INTERPRETABILITY ANALYSIS:
{shap_insights}
{lime_insights}

STUDENT DATA:
{features_string}

INSTRUCTIONS:
You must provide a comprehensive, data-driven interpretation that combines the quantitative interpretability results with domain expertise in education. Your analysis should be:

1. QUANTITATIVELY INFORMED: Use the SHAP values and LIME explanations to identify the most influential features
2. EDUCATIONALLY CONTEXTUALIZED: Explain how each significant feature relates to student success in educational terms
3. ACTIONABLE: Provide specific, evidence-based recommendations for improvement
4. UNCERTAINTY-AWARE: Acknowledge limitations and areas where more data might be needed

Structure your response as follows:

EXECUTIVE SUMMARY:
[Provide a clear, one-paragraph summary of the prediction and key contributing factors]

QUANTITATIVE FEATURE ANALYSIS:
[For each of the top 3-4 features identified by SHAP/LIME, provide detailed analysis explaining:
- The specific contribution value and direction
- Why this feature matters educationally
- How the student's specific value compares to typical patterns
- Specific recommendations related to this feature]

MODEL CONSENSUS ANALYSIS:
[Analyze agreement/disagreement across models and what this tells us about prediction confidence]

ACTIONABLE RECOMMENDATIONS:
[Provide 3-5 specific, evidence-based recommendations for improving student outcomes]

UNCERTAINTY AND LIMITATIONS:
[Discuss what the model might be missing and where human judgment is still crucial]

Begin your evidence-based interpretation now:
'''
        return prompt

def create_model_interpreter(models_dict: Dict, training_data: pd.DataFrame,
                           feature_names: List[str], feature_descriptions: Dict[str, str],
                           categorical_features: List[str], preprocessors: Dict = None) -> ModelInterpreter:
    """
    Factory function to create a ModelInterpreter instance.

    Args:
        models_dict: Dictionary of trained models
        training_data: Training dataset
        feature_names: List of feature names
        feature_descriptions: Dictionary of feature descriptions
        categorical_features: List of categorical feature names
        preprocessors: Dictionary of preprocessing objects

    Returns:
        ModelInterpreter instance
    """
    return ModelInterpreter(
        models_dict, training_data, feature_names,
        feature_descriptions, categorical_features, preprocessors
    )

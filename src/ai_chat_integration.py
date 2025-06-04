"""
AI Chat Integration Module for Hayzed Flask App
==============================================

This module integrates the enhanced AI chat system with the existing Flask application,
providing seamless backward compatibility while enabling advanced features.
"""

import os
import uuid
import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from flask import session, request, jsonify

from enhanced_ai_chat import (
    EnhancedAIChat,
    create_enhanced_ai_chat_instance,
    format_prediction_data_for_enhanced_chat,
    ExplanationMode,
    ResponseStyle,
    QueryComplexity
)


class AIIntegrationManager:
    """
    Manages the integration between the enhanced AI chat and Flask app.
    """

    def __init__(self, groq_api_key: str):
        """Initialize the AI integration manager."""
        self.enhanced_chat = create_enhanced_ai_chat_instance(groq_api_key)
        self.fallback_enabled = True

    def get_session_id(self) -> str:
        """Get or create a session ID for the current user."""
        if 'ai_session_id' not in session:
            session['ai_session_id'] = str(uuid.uuid4())
        return session['ai_session_id']

    def detect_user_preferences_from_query(self, query: str) -> Dict[str, Any]:
        """
        Detect user preferences from their query to auto-adapt the experience.
        """
        query_lower = query.lower()
        preferences = {}

        # Detect expertise level
        if any(word in query_lower for word in ['simple', 'basic', 'explain like', 'beginner']):
            preferences['expertise_level'] = 'beginner'
        elif any(word in query_lower for word in ['technical', 'detailed', 'statistical', 'advanced']):
            preferences['expertise_level'] = 'advanced'
        elif any(word in query_lower for word in ['research', 'academic', 'methodology', 'algorithm']):
            preferences['expertise_level'] = 'expert'

        # Detect response style preferences
        if any(word in query_lower for word in ['summary', 'brief', 'short', 'quick']):
            preferences['response_style'] = 'summary'
        elif any(word in query_lower for word in ['formal', 'report', 'academic', 'professional']):
            preferences['response_style'] = 'formal'
        elif any(word in query_lower for word in ['teach', 'learn', 'understand', 'educational']):
            preferences['response_style'] = 'educational'
        elif any(word in query_lower for word in ['technical', 'methodology', 'algorithm']):
            preferences['response_style'] = 'technical'

        # Detect features of interest
        feature_keywords = {
            'academic': ['grade', 'study', 'academic', 'performance', 'exam'],
            'social': ['social', 'friends', 'family', 'relationship'],
            'behavioral': ['behavior', 'attendance', 'absence', 'participation'],
            'demographic': ['age', 'gender', 'location', 'background'],
            'lifestyle': ['alcohol', 'health', 'activities', 'freetime']
        }

        features_of_interest = []
        for category, keywords in feature_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                features_of_interest.append(category)

        if features_of_interest:
            preferences['features_of_interest'] = features_of_interest

        return preferences

    def get_enhanced_ai_interpretation(
        self,
        features: Dict[str, Any],
        feature_descriptions: Dict[str, str],
        all_feature_names: List[str],
        prediction_outcome: str,
        all_model_preds: Dict[str, str],
        interpretability_data: Optional[Dict[str, Any]] = None,
        custom_query: Optional[str] = None
    ) -> Tuple[str, List[Dict[str, str]], Dict[str, Any]]:
        """
        Enhanced version of get_ai_interpretation with backward compatibility.

        Returns:
            Tuple of (interpretation_text, conversation_history, additional_data)
        """
        session_id = self.get_session_id()

        # Format prediction data for enhanced chat
        prediction_data = format_prediction_data_for_enhanced_chat(
            features=features,
            feature_descriptions=feature_descriptions,
            model_predictions=all_model_preds,
            primary_prediction=prediction_outcome,
            interpretability_data=convert_numpy_types(interpretability_data) if interpretability_data else None
        )

        # Apply comprehensive numpy conversion to all prediction data
        prediction_data = convert_numpy_types(prediction_data)

        # Use custom query or create default analysis request
        if custom_query:
            query = custom_query
        else:
            query = "Please provide a comprehensive analysis of this student's performance prediction, including key factors influencing the outcome and actionable insights."

        # Detect user preferences from query
        user_preferences = self.detect_user_preferences_from_query(query)

        # Enhanced AI not available, fall back to basic implementation
        if not self.enhanced_chat:
            return self._fallback_to_basic_ai(
                features, feature_descriptions, all_feature_names,
                prediction_outcome, all_model_preds
            )

        try:
            # Get enhanced response synchronously (Flask doesn't support async easily)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            ai_response = loop.run_until_complete(
                self.enhanced_chat.get_enhanced_response(
                    session_id=session_id,
                    query=query,
                    prediction_data=prediction_data,
                    is_follow_up=False,
                    user_preferences=user_preferences
                )
            )

            loop.close()

            # Convert to format expected by existing Flask routes
            conversation_history = [
                {"role": "user", "content": query},
                {"role": "assistant", "content": ai_response.content}
            ]

            # Additional data for enhanced features - ensure JSON serializable
            additional_data = {
                "model_used": ai_response.model_used,
                "confidence_score": float(ai_response.confidence_score),
                "response_time": float(ai_response.response_time),
                "suggested_follow_ups": ai_response.suggested_follow_ups,
                "explanation_level": ai_response.explanation_level.value,
                "tokens_used": int(ai_response.tokens_used)
            }

            return ai_response.content, conversation_history, additional_data

        except Exception as e:
            print(f"Enhanced AI failed, falling back to basic: {str(e)}")
            return self._fallback_to_basic_ai(
                features, feature_descriptions, all_feature_names,
                prediction_outcome, all_model_preds
            )

    def get_enhanced_follow_up_response(
        self,
        follow_up_message: str,
        conversation_history: List[Dict[str, str]],
        features: Dict[str, Any],
        feature_descriptions: Dict[str, str],
        all_model_preds: Dict[str, str],
        prediction_outcome: str
    ) -> Tuple[str, List[Dict[str, str]], Dict[str, Any]]:
        """
        Enhanced follow-up response with intelligent context management.
        """
        session_id = self.get_session_id()

        if not self.enhanced_chat:
            return self._fallback_follow_up(follow_up_message, conversation_history)

        try:
            # Format prediction data
            prediction_data = format_prediction_data_for_enhanced_chat(
                features=features,
                feature_descriptions=feature_descriptions,
                model_predictions=all_model_preds,
                primary_prediction=prediction_outcome
            )

            # Apply comprehensive numpy conversion to all prediction data
            prediction_data = convert_numpy_types(prediction_data)

            # Detect preferences from follow-up message
            user_preferences = self.detect_user_preferences_from_query(follow_up_message)

            # Get enhanced follow-up response
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            ai_response = loop.run_until_complete(
                self.enhanced_chat.get_enhanced_response(
                    session_id=session_id,
                    query=follow_up_message,
                    prediction_data=prediction_data,
                    is_follow_up=True,
                    user_preferences=user_preferences
                )
            )

            loop.close()

            # Get updated conversation history from enhanced chat
            updated_history = self.enhanced_chat.conversation_histories.get(session_id, [])

            # Additional data
            additional_data = {
                "model_used": ai_response.model_used,
                "confidence_score": float(ai_response.confidence_score),
                "response_time": float(ai_response.response_time),
                "suggested_follow_ups": ai_response.suggested_follow_ups,
                "explanation_level": ai_response.explanation_level.value,
                "tokens_used": int(ai_response.tokens_used)
            }

            return ai_response.content, updated_history, additional_data

        except Exception as e:
            print(f"Enhanced follow-up failed, falling back: {str(e)}")
            return self._fallback_follow_up(follow_up_message, conversation_history)

    def _fallback_to_basic_ai(
        self,
        features: Dict[str, Any],
        feature_descriptions: Dict[str, str],
        all_feature_names: List[str],
        prediction_outcome: str,
        all_model_preds: Dict[str, str]
    ) -> Tuple[str, List[Dict[str, str]], Dict[str, Any]]:
        """Fallback to the original AI implementation."""

        # Import the original function to maintain compatibility
        from app import get_ai_interpretation, custom_ai_prompt

        # Create the original prompt
        prompt = custom_ai_prompt(features, feature_descriptions, all_feature_names, prediction_outcome, all_model_preds)
        conversation_history = [{"role": "user", "content": prompt}]

        # Call original function
        ai_response, updated_history = get_ai_interpretation(
            input_features=features,
            feature_descriptions=feature_descriptions,
            all_feature_names=all_feature_names,
            prediction_outcome=prediction_outcome,
            conversation_history=conversation_history
        )

        additional_data = {
            "model_used": "llama-3.1-8b-instant",  # Default from original
            "confidence_score": 0.7,  # Default estimate
            "response_time": 0.0,
            "suggested_follow_ups": [],
            "explanation_level": "intermediate",
            "tokens_used": 0
        }

        return ai_response, updated_history, additional_data

    def _fallback_follow_up(
        self,
        follow_up_message: str,
        conversation_history: List[Dict[str, str]]
    ) -> Tuple[str, List[Dict[str, str]], Dict[str, Any]]:
        """Fallback for follow-up messages."""

        from app import get_ai_interpretation

        ai_response, updated_history = get_ai_interpretation(
            input_features={},  # Not used in follow-up
            feature_descriptions={},
            all_feature_names=[],
            conversation_history=conversation_history,
            follow_up_message=follow_up_message
        )

        additional_data = {
            "model_used": "llama-3.1-8b-instant",
            "confidence_score": 0.7,
            "response_time": 0.0,
            "suggested_follow_ups": [],
            "explanation_level": "intermediate",
            "tokens_used": 0
        }

        return ai_response, updated_history, additional_data

    def get_user_preferences_ui_data(self) -> Dict[str, Any]:
        """Get data for user preferences UI."""
        return {
            "expertise_levels": [
                {"value": "beginner", "label": "Beginner - Simple explanations", "description": "Clear, jargon-free explanations with practical focus"},
                {"value": "intermediate", "label": "Intermediate - Balanced detail", "description": "Mix of practical insights and technical context"},
                {"value": "advanced", "label": "Advanced - Technical detail", "description": "Detailed methodology with statistical context"},
                {"value": "expert", "label": "Expert - Full technical rigor", "description": "Comprehensive analysis with research-level detail"}
            ],
            "response_styles": [
                {"value": "conversational", "label": "Conversational", "description": "Friendly and engaging tone"},
                {"value": "formal", "label": "Formal Report", "description": "Academic style with structured headings"},
                {"value": "technical", "label": "Technical", "description": "Focus on methodology and metrics"},
                {"value": "educational", "label": "Educational", "description": "Teaching-focused explanations"},
                {"value": "summary", "label": "Summary", "description": "Concise key insights only"}
            ],
            "feature_categories": [
                {"value": "academic", "label": "Academic Performance", "description": "Grades, study habits, exam performance"},
                {"value": "social", "label": "Social Factors", "description": "Family, friends, relationships"},
                {"value": "behavioral", "label": "Behavioral Patterns", "description": "Attendance, participation, engagement"},
                {"value": "demographic", "label": "Demographics", "description": "Age, gender, background factors"},
                {"value": "lifestyle", "label": "Lifestyle Factors", "description": "Health, activities, free time"}
            ]
        }

    def update_user_preferences(self, preferences: Dict[str, Any]) -> bool:
        """Update user preferences for the current session."""
        session_id = self.get_session_id()

        if self.enhanced_chat:
            return self.enhanced_chat.update_user_preferences(session_id, preferences)

        # Store in Flask session as fallback
        session['ai_user_preferences'] = preferences
        return True

    def get_session_insights(self) -> Dict[str, Any]:
        """Get insights about the current chat session."""
        session_id = self.get_session_id()

        if self.enhanced_chat:
            return self.enhanced_chat.get_session_summary(session_id)

        # Basic fallback
        return {
            "session_id": session_id,
            "total_interactions": session.get('ai_interaction_count', 0),
            "preferences": session.get('ai_user_preferences', {})
        }

    def clear_session_data(self) -> None:
        """Clear all AI session data."""
        session_id = self.get_session_id()

        if self.enhanced_chat:
            self.enhanced_chat.clear_session(session_id)

        # Clear Flask session data
        keys_to_remove = [key for key in session.keys() if key.startswith('ai_')]
        for key in keys_to_remove:
            session.pop(key, None)


# Utility functions for easy integration with existing Flask routes
def create_ai_integration_manager(groq_api_key: str) -> Optional[AIIntegrationManager]:
    """Create an AI integration manager instance."""
    try:
        return AIIntegrationManager(groq_api_key)
    except Exception as e:
        print(f"Failed to create AI integration manager: {str(e)}")
        return None


def enhance_interpretation_route_response(
    interpretation_text: str,
    conversation_history: List[Dict[str, str]],
    additional_data: Dict[str, Any],
    base_response: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Enhance the response from interpretation routes with additional AI data.
    """
    enhanced_response = base_response.copy()

    # Add enhanced AI data
    enhanced_response.update({
        "interpretation": interpretation_text,
        "conversation_history": conversation_history,
        "ai_metadata": {
            "model_used": additional_data.get("model_used"),
            "confidence_score": additional_data.get("confidence_score"),
            "response_time": additional_data.get("response_time"),
            "explanation_level": additional_data.get("explanation_level"),
            "tokens_used": additional_data.get("tokens_used")
        },
        "suggested_follow_ups": additional_data.get("suggested_follow_ups", []),
        "enhanced_features_available": True
    })

    return enhanced_response


def enhance_follow_up_route_response(
    reply_text: str,
    conversation_history: List[Dict[str, str]],
    additional_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Enhance the response from follow-up routes with additional AI data.
    """
    return {
        "reply": reply_text,
        "history": conversation_history,
        "ai_metadata": {
            "model_used": additional_data.get("model_used"),
            "confidence_score": additional_data.get("confidence_score"),
            "response_time": additional_data.get("response_time"),
            "explanation_level": additional_data.get("explanation_level"),
            "tokens_used": additional_data.get("tokens_used")
        },
        "suggested_follow_ups": additional_data.get("suggested_follow_ups", []),
        "enhanced_features_available": True
    }


def create_smart_follow_up_suggestions(
    content: str,
    prediction_data: Dict[str, Any],
    user_context: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Create intelligent follow-up suggestions based on content and context.
    """
    suggestions = []
    content_lower = content.lower()

    # Content-based suggestions
    if 'important' in content_lower and 'feature' in content_lower:
        suggestions.append("How do these important features interact with each other?")

    if 'prediction' in content_lower and ('confident' in content_lower or 'accuracy' in content_lower):
        suggestions.append("What factors could make this prediction more or less reliable?")

    if 'improve' in content_lower or 'recommendation' in content_lower:
        suggestions.append("What would be the most effective way to implement these improvements?")

    # Prediction-specific suggestions
    primary_prediction = prediction_data.get('primary_prediction', '').lower()
    if 'pass' in primary_prediction:
        suggestions.append("What factors are most critical for maintaining this positive trajectory?")
    elif 'fail' in primary_prediction:
        suggestions.append("What interventions could most effectively change this outcome?")

    # Context-based suggestions
    if user_context:
        expertise = user_context.get('expertise_level', 'intermediate')
        if expertise == 'beginner':
            suggestions.append("Can you explain this in even simpler terms?")
        elif expertise in ['advanced', 'expert']:
            suggestions.append("What are the methodological limitations of this analysis?")

    # Ensure we have enough variety
    default_suggestions = [
        "Tell me more about the model's decision-making process",
        "How does this student compare to typical patterns?",
        "What additional data would improve this prediction?",
        "Can you focus on one specific aspect in more detail?"
    ]

    # Combine and deduplicate
    all_suggestions = suggestions + default_suggestions
    unique_suggestions = []
    seen = set()

    for suggestion in all_suggestions:
        if suggestion.lower() not in seen:
            unique_suggestions.append(suggestion)
            seen.add(suggestion.lower())

    return unique_suggestions[:4]  # Return top 4 suggestions


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    """
    if hasattr(obj, 'dtype'):  # numpy arrays and scalars
        if obj.dtype.kind in ['i', 'u']:  # integer types
            return int(obj) if obj.ndim == 0 else obj.astype(int).tolist()
        elif obj.dtype.kind == 'f':  # floating point types
            return float(obj) if obj.ndim == 0 else obj.astype(float).tolist()
        elif obj.dtype.kind == 'b':  # boolean types
            return bool(obj) if obj.ndim == 0 else obj.astype(bool).tolist()
        else:
            return obj.tolist() if hasattr(obj, 'tolist') else str(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

"""
Enhanced AI Chat System for Hayzed Student Performance Prediction
=================================================================

This module provides an advanced AI chat system that significantly improves upon
the basic Groq integration with features like:

- Dynamic model selection based on query complexity
- Advanced prompt engineering with context awareness
- Optimized conversation memory management
- Adaptive response strategies
- Enhanced error handling and fallback mechanisms
- User preference adaptation
- Multi-modal explanations
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import re

from groq import Groq
import numpy as np


class QueryComplexity(Enum):
    """Enum to categorize query complexity for model selection."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    EXPERT = "expert"


class ResponseStyle(Enum):
    """Enum for different response styles."""
    FORMAL = "formal"
    CONVERSATIONAL = "conversational"
    TECHNICAL = "technical"
    EDUCATIONAL = "educational"
    SUMMARY = "summary"


class ExplanationMode(Enum):
    """Enum for different explanation modes."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class ChatContext:
    """Enhanced context for chat conversations."""
    user_expertise_level: ExplanationMode = ExplanationMode.INTERMEDIATE
    preferred_style: ResponseStyle = ResponseStyle.CONVERSATIONAL
    conversation_focus: str = "general"
    features_of_interest: List[str] = None
    previous_queries: List[str] = None
    session_start_time: datetime = None
    total_interactions: int = 0

    def __post_init__(self):
        if self.features_of_interest is None:
            self.features_of_interest = []
        if self.previous_queries is None:
            self.previous_queries = []
        if self.session_start_time is None:
            self.session_start_time = datetime.now()


@dataclass
class AIResponse:
    """Enhanced AI response object."""
    content: str
    confidence_score: float
    model_used: str
    response_time: float
    tokens_used: int
    suggested_follow_ups: List[str] = None
    citations: List[str] = None
    explanation_level: ExplanationMode = ExplanationMode.INTERMEDIATE

    def __post_init__(self):
        if self.suggested_follow_ups is None:
            self.suggested_follow_ups = []
        if self.citations is None:
            self.citations = []


class EnhancedAIChat:
    """
    Enhanced AI Chat system with advanced features for student performance prediction.
    """

    # Available Groq models with their characteristics
    GROQ_MODELS = {
        "llama-3.1-8b-instant": {
            "max_tokens": 8192,
            "speed": "fastest",
            "complexity": [QueryComplexity.SIMPLE, QueryComplexity.MEDIUM],
            "cost": "low"
        },
        "llama-3.1-70b-versatile": {
            "max_tokens": 8192,
            "speed": "fast",
            "complexity": [QueryComplexity.MEDIUM, QueryComplexity.COMPLEX],
            "cost": "medium"
        },
        "llama-3.3-70b-versatile": {
            "max_tokens": 32768,
            "speed": "moderate",
            "complexity": [QueryComplexity.COMPLEX, QueryComplexity.EXPERT],
            "cost": "high"
        },
        "mixtral-8x7b-32768": {
            "max_tokens": 32768,
            "speed": "fast",
            "complexity": [QueryComplexity.MEDIUM, QueryComplexity.COMPLEX],
            "cost": "medium"
        }
    }

    def __init__(self, api_key: str, default_model: str = "llama-3.1-70b-versatile"):
        """
        Initialize the Enhanced AI Chat system.

        Args:
            api_key: Groq API key
            default_model: Default model to use if auto-selection fails
        """
        if not api_key:
            raise ValueError("Groq API key is required")

        self.client = Groq(api_key=api_key)
        self.default_model = default_model
        self.conversation_histories = {}  # session_id -> conversation history
        self.chat_contexts = {}  # session_id -> ChatContext
        self.response_cache = {}  # Simple cache for repeated queries

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def analyze_query_complexity(self, query: str, context: Optional[ChatContext] = None) -> QueryComplexity:
        """
        Analyze the complexity of a user query to select appropriate model.

        Args:
            query: User's query
            context: Current chat context

        Returns:
            QueryComplexity level
        """
        query_lower = query.lower()

        # Keywords that indicate different complexity levels
        simple_keywords = [
            "what is", "explain", "simple", "basic", "overview", "summary",
            "yes", "no", "thanks", "hello", "hi"
        ]

        medium_keywords = [
            "how does", "why did", "compare", "analyze", "relationship",
            "correlation", "impact", "influence", "factor"
        ]

        complex_keywords = [
            "statistical significance", "confidence interval", "feature importance",
            "model performance", "cross-validation", "overfitting", "bias-variance",
            "ensemble", "hyperparameter", "regularization"
        ]

        expert_keywords = [
            "bayesian", "monte carlo", "gradient descent", "backpropagation",
            "shap values", "lime explanation", "adversarial", "interpretability",
            "causality", "confounding", "heteroscedasticity"
        ]

        # Count keyword matches
        simple_score = sum(1 for keyword in simple_keywords if keyword in query_lower)
        medium_score = sum(1 for keyword in medium_keywords if keyword in query_lower)
        complex_score = sum(1 for keyword in complex_keywords if keyword in query_lower)
        expert_score = sum(1 for keyword in expert_keywords if keyword in query_lower)

        # Consider query length and structure
        word_count = len(query.split())

        if expert_score > 0 or (context and context.user_expertise_level == ExplanationMode.EXPERT):
            return QueryComplexity.EXPERT
        elif complex_score > 0 or word_count > 50:
            return QueryComplexity.COMPLEX
        elif medium_score > 0 or word_count > 20:
            return QueryComplexity.MEDIUM
        else:
            return QueryComplexity.SIMPLE

    def select_optimal_model(self, complexity: QueryComplexity, context: Optional[ChatContext] = None) -> str:
        """
        Select the optimal model based on query complexity and context.

        Args:
            complexity: Query complexity level
            context: Current chat context

        Returns:
            Selected model name
        """
        suitable_models = []

        for model_name, model_info in self.GROQ_MODELS.items():
            if complexity in model_info["complexity"]:
                suitable_models.append((model_name, model_info))

        if not suitable_models:
            return self.default_model

        # For now, select the first suitable model
        # In the future, could add cost optimization, user preferences, etc.
        return suitable_models[0][0]

    def create_adaptive_prompt(
        self,
        query: str,
        prediction_data: Dict[str, Any],
        context: ChatContext,
        is_follow_up: bool = False
    ) -> str:
        """
        Create an adaptive prompt based on context and user preferences.

        Args:
            query: User's query
            prediction_data: Prediction results and feature data
            context: Current chat context
            is_follow_up: Whether this is a follow-up question

        Returns:
            Formatted prompt string
        """
        # Base system prompt that adapts based on context
        system_instructions = self._generate_system_instructions(context, is_follow_up)

        if is_follow_up:
            # For follow-ups, create a concise prompt focusing on the specific question
            prompt = f"{system_instructions}\n\nUser's follow-up question: {query}\n\nPlease provide a direct, helpful answer."
        else:
            # For initial queries, create comprehensive prompt
            prompt = self._generate_comprehensive_prompt(query, prediction_data, context)

        return prompt

    def _generate_system_instructions(self, context: ChatContext, is_follow_up: bool = False) -> str:
        """Generate adaptive system instructions based on context."""

        # Base personality
        base_instruction = "You are an expert AI assistant specializing in student performance prediction analysis."

        # Adapt based on expertise level
        expertise_instructions = {
            ExplanationMode.BEGINNER: "Explain concepts in simple terms, avoid jargon, and use analogies when helpful. Focus on practical insights.",
            ExplanationMode.INTERMEDIATE: "Provide balanced explanations with some technical detail but keep it accessible. Include both practical insights and methodological context.",
            ExplanationMode.ADVANCED: "Use appropriate technical terminology and provide detailed methodological insights. Include statistical context and model-specific details.",
            ExplanationMode.EXPERT: "Provide comprehensive technical analysis with full statistical and methodological rigor. Include advanced interpretability techniques and research context."
        }

        # Adapt based on response style
        style_instructions = {
            ResponseStyle.FORMAL: "Use formal academic language and structure your response with clear headings and systematic analysis.",
            ResponseStyle.CONVERSATIONAL: "Use a friendly, conversational tone while maintaining professionalism. Make the analysis engaging and approachable.",
            ResponseStyle.TECHNICAL: "Focus on technical accuracy and detailed methodology. Include relevant metrics, statistical tests, and model specifications.",
            ResponseStyle.EDUCATIONAL: "Structure your response as a learning experience. Explain not just what the results show, but why they matter and how to interpret them.",
            ResponseStyle.SUMMARY: "Provide concise, well-organized summaries focusing on key insights and actionable recommendations."
        }

        expertise_instruction = expertise_instructions.get(context.user_expertise_level, expertise_instructions[ExplanationMode.INTERMEDIATE])
        style_instruction = style_instructions.get(context.preferred_style, style_instructions[ResponseStyle.CONVERSATIONAL])

        system_instruction = f"""{base_instruction}

EXPERTISE LEVEL: {expertise_instruction}

RESPONSE STYLE: {style_instruction}"""

        if is_follow_up:
            system_instruction += "\n\nThis is a follow-up question. Provide a direct, focused answer that builds on the previous conversation. Do not repeat the full analysis unless specifically requested."

        if context.features_of_interest:
            system_instruction += f"\n\nUSER'S AREAS OF INTEREST: Pay special attention to these features: {', '.join(context.features_of_interest)}"

        return system_instruction

    def _generate_comprehensive_prompt(self, query: str, prediction_data: Dict[str, Any], context: ChatContext) -> str:
        """Generate a comprehensive prompt for initial analysis."""

        # Extract data components
        features = prediction_data.get('features', {})
        feature_descriptions = prediction_data.get('feature_descriptions', {})
        model_predictions = prediction_data.get('model_predictions', {})
        primary_prediction = prediction_data.get('primary_prediction', 'Unknown')
        interpretability_data = prediction_data.get('interpretability_data', {})

        # Build feature information string
        features_string = ""
        for feature_name, value in features.items():
            description = feature_descriptions.get(feature_name, "No description available.")
            features_string += f"• {feature_name.replace('_', ' ').title()}: {value}\n  Context: {description}\n\n"

        # Build model predictions string
        model_preds_str = ""
        for model_name, prediction in model_predictions.items():
            model_preds_str += f"• {model_name}: {prediction}\n"

        # Include interpretability insights if available
        interpretability_section = ""
        if interpretability_data:
            interpretability_section = f"""
ADVANCED INTERPRETABILITY INSIGHTS:
{self._format_interpretability_data(interpretability_data, context)}
"""

        # Adaptive prompt based on expertise level
        analysis_depth = {
            ExplanationMode.BEGINNER: "Focus on practical insights and actionable recommendations. Explain what the prediction means in simple terms.",
            ExplanationMode.INTERMEDIATE: "Provide balanced analysis including key factors, model insights, and practical implications.",
            ExplanationMode.ADVANCED: "Include detailed feature analysis, model comparison, statistical insights, and methodological considerations.",
            ExplanationMode.EXPERT: "Provide comprehensive technical analysis including advanced interpretability, statistical significance, model limitations, and research implications."
        }

        depth_instruction = analysis_depth.get(context.user_expertise_level, analysis_depth[ExplanationMode.INTERMEDIATE])

        prompt = f"""STUDENT PERFORMANCE PREDICTION ANALYSIS

PREDICTION RESULTS:
{model_preds_str}
Primary Prediction (Best Model): {primary_prediction}

STUDENT DATA:
{features_string}

{interpretability_section}

ANALYSIS REQUEST: {query}

ANALYSIS DEPTH: {depth_instruction}

Please provide a comprehensive analysis that addresses the user's specific request while incorporating all available information. Structure your response clearly and make it actionable."""

        return prompt

    def _format_interpretability_data(self, interpretability_data: Dict[str, Any], context: ChatContext) -> str:
        """Format interpretability data based on user expertise level."""

        formatted_sections = []

        # SHAP values formatting
        if 'shap_values' in interpretability_data:
            shap_section = "SHAP Analysis:\n"
            shap_values = interpretability_data['shap_values']

            if context.user_expertise_level in [ExplanationMode.BEGINNER, ExplanationMode.INTERMEDIATE]:
                shap_section += "Feature impact on prediction (positive values increase likelihood, negative values decrease it):\n"
            else:
                shap_section += "SHAP feature attribution values (marginal contributions to log-odds):\n"

            for feature, value in shap_values.items():
                shap_section += f"• {feature}: {value:.4f}\n"

            formatted_sections.append(shap_section)

        # Add other interpretability sections as needed
        # (feature interactions, partial dependence, etc.)

        return "\n".join(formatted_sections)

    def optimize_conversation_history(self, session_id: str, max_context_length: int = 4000) -> List[Dict[str, str]]:
        """
        Optimize conversation history to stay within token limits while preserving context.

        Args:
            session_id: Session identifier
            max_context_length: Maximum character length for context

        Returns:
            Optimized conversation history
        """
        if session_id not in self.conversation_histories:
            return []

        history = self.conversation_histories[session_id]

        if not history:
            return []

        # Calculate current length
        current_length = sum(len(msg.get('content', '')) for msg in history)

        if current_length <= max_context_length:
            return history

        # Keep the most recent messages and summarize older ones
        optimized_history = []
        current_size = 0

        # Always keep the last few exchanges
        recent_messages = history[-6:]  # Last 3 user-assistant pairs

        for msg in reversed(recent_messages):
            content_length = len(msg.get('content', ''))
            if current_size + content_length <= max_context_length:
                optimized_history.insert(0, msg)
                current_size += content_length
            else:
                break

        # If we have space, add a summary of earlier conversation
        if current_size < max_context_length * 0.8 and len(history) > len(optimized_history):
            earlier_messages = history[:-len(optimized_history)]
            summary = self._summarize_conversation_segment(earlier_messages)

            summary_msg = {
                "role": "system",
                "content": f"Previous conversation summary: {summary}"
            }

            if len(summary) < max_context_length - current_size:
                optimized_history.insert(0, summary_msg)

        return optimized_history

    def _summarize_conversation_segment(self, messages: List[Dict[str, str]]) -> str:
        """Summarize a segment of conversation history."""
        if not messages:
            return ""

        # Extract key topics and user interests
        user_messages = [msg['content'] for msg in messages if msg['role'] == 'user']

        # Simple extractive summary (could be enhanced with AI summarization)
        key_topics = []
        for msg in user_messages:
            # Extract question words and main topics
            words = msg.lower().split()
            question_indicators = ['what', 'why', 'how', 'when', 'where', 'which']
            for i, word in enumerate(words):
                if word in question_indicators and i < len(words) - 1:
                    # Take next few words as topic
                    topic = ' '.join(words[i:i+4])
                    key_topics.append(topic)

        if key_topics:
            return f"User previously asked about: {'; '.join(key_topics[:3])}"
        else:
            return "User had previous questions about the prediction analysis"

    def generate_follow_up_suggestions(self, response_content: str, context: ChatContext) -> List[str]:
        """
        Generate intelligent follow-up question suggestions.

        Args:
            response_content: The AI's response content
            context: Current chat context

        Returns:
            List of suggested follow-up questions
        """
        suggestions = []

        # Content-based suggestions
        content_lower = response_content.lower()

        if 'feature' in content_lower and 'important' in content_lower:
            suggestions.append("Can you explain how these important features interact with each other?")

        if 'model' in content_lower and ('predict' in content_lower or 'performance' in content_lower):
            suggestions.append("How confident should I be in this prediction?")

        if 'improve' in content_lower or 'recommendation' in content_lower:
            suggestions.append("What specific actions would have the biggest impact?")

        # Context-based suggestions
        if context.user_expertise_level == ExplanationMode.BEGINNER:
            suggestions.extend([
                "Can you explain this in simpler terms?",
                "What does this mean in practical terms?",
                "Are there any common misconceptions about this?"
            ])
        elif context.user_expertise_level == ExplanationMode.EXPERT:
            suggestions.extend([
                "What are the statistical assumptions underlying this analysis?",
                "How would different model architectures affect these results?",
                "What are the potential confounding factors?"
            ])

        # Feature-specific suggestions
        if context.features_of_interest:
            feature = context.features_of_interest[0].replace('_', ' ')
            suggestions.append(f"Tell me more about how {feature} specifically impacts the prediction")

        # Return top 3-4 suggestions
        return suggestions[:4]

    async def get_enhanced_response(
        self,
        session_id: str,
        query: str,
        prediction_data: Dict[str, Any],
        is_follow_up: bool = False,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> AIResponse:
        """
        Get an enhanced AI response with all advanced features.

        Args:
            session_id: Unique session identifier
            query: User's query
            prediction_data: Prediction results and feature data
            is_follow_up: Whether this is a follow-up question
            user_preferences: Optional user preferences override

        Returns:
            AIResponse object with enhanced information
        """
        start_time = time.time()

        # Initialize or get context
        if session_id not in self.chat_contexts:
            self.chat_contexts[session_id] = ChatContext()

        context = self.chat_contexts[session_id]

        # Update context with user preferences if provided
        if user_preferences:
            if 'expertise_level' in user_preferences:
                context.user_expertise_level = ExplanationMode(user_preferences['expertise_level'])
            if 'response_style' in user_preferences:
                context.preferred_style = ResponseStyle(user_preferences['response_style'])
            if 'features_of_interest' in user_preferences:
                context.features_of_interest = user_preferences['features_of_interest']

        # Update interaction tracking
        context.total_interactions += 1
        context.previous_queries.append(query)

        # Analyze query complexity and select model
        complexity = self.analyze_query_complexity(query, context)
        selected_model = self.select_optimal_model(complexity, context)

        # Create adaptive prompt
        prompt = self.create_adaptive_prompt(query, prediction_data, context, is_follow_up)

        # Prepare conversation history
        if session_id not in self.conversation_histories:
            self.conversation_histories[session_id] = []

        conversation_history = self.optimize_conversation_history(session_id)

        # Prepare messages for API call
        messages = []

        if is_follow_up:
            messages.extend(conversation_history)
            messages.append({"role": "user", "content": query})
        else:
            messages.append({"role": "user", "content": prompt})

        # Determine temperature based on query type and complexity
        temperature = self._get_adaptive_temperature(complexity, context.preferred_style, is_follow_up)

        try:
            # Make API call
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=selected_model,
                temperature=temperature,
                max_tokens=min(2048, self.GROQ_MODELS[selected_model]["max_tokens"] // 2),
                top_p=0.95,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )

            response_content = chat_completion.choices[0].message.content
            response_time = time.time() - start_time

            # Update conversation history
            if is_follow_up:
                self.conversation_histories[session_id].append({"role": "user", "content": query})
            else:
                self.conversation_histories[session_id] = [{"role": "user", "content": prompt}]

            self.conversation_histories[session_id].append({"role": "assistant", "content": response_content})

            # Generate follow-up suggestions
            follow_up_suggestions = self.generate_follow_up_suggestions(response_content, context)

            # Calculate confidence score (simple heuristic)
            confidence_score = self._calculate_confidence_score(response_content, complexity, selected_model)

            # Create enhanced response
            ai_response = AIResponse(
                content=response_content,
                confidence_score=confidence_score,
                model_used=selected_model,
                response_time=response_time,
                tokens_used=chat_completion.usage.total_tokens if hasattr(chat_completion, 'usage') else 0,
                suggested_follow_ups=follow_up_suggestions,
                explanation_level=context.user_expertise_level
            )

            self.logger.info(f"Generated response for session {session_id} using {selected_model} in {response_time:.2f}s")

            return ai_response

        except Exception as e:
            self.logger.error(f"Error generating AI response: {str(e)}")

            # Fallback response
            fallback_content = self._generate_fallback_response(query, is_follow_up)

            return AIResponse(
                content=fallback_content,
                confidence_score=0.0,
                model_used="fallback",
                response_time=time.time() - start_time,
                tokens_used=0,
                suggested_follow_ups=["Can you rephrase your question?", "Would you like to try a different approach?"]
            )

    def _get_adaptive_temperature(
        self,
        complexity: QueryComplexity,
        style: ResponseStyle,
        is_follow_up: bool
    ) -> float:
        """Get adaptive temperature based on context."""

        base_temperatures = {
            QueryComplexity.SIMPLE: 0.3,
            QueryComplexity.MEDIUM: 0.5,
            QueryComplexity.COMPLEX: 0.7,
            QueryComplexity.EXPERT: 0.8
        }

        style_adjustments = {
            ResponseStyle.FORMAL: -0.1,
            ResponseStyle.CONVERSATIONAL: 0.0,
            ResponseStyle.TECHNICAL: -0.2,
            ResponseStyle.EDUCATIONAL: 0.1,
            ResponseStyle.SUMMARY: -0.1
        }

        temperature = base_temperatures.get(complexity, 0.5)
        temperature += style_adjustments.get(style, 0.0)

        if is_follow_up:
            temperature -= 0.1  # More focused for follow-ups

        return max(0.1, min(1.0, temperature))

    def _calculate_confidence_score(self, content: str, complexity: QueryComplexity, model: str) -> float:
        """Calculate a confidence score for the response."""

        # Simple heuristic based on response characteristics
        score = 0.7  # Base score

        # Length-based adjustment
        word_count = len(content.split())
        if 50 <= word_count <= 300:
            score += 0.1
        elif word_count < 20:
            score -= 0.2

        # Model capability adjustment
        model_scores = {
            "llama-3.3-70b-versatile": 0.9,
            "llama-3.1-70b-versatile": 0.8,
            "mixtral-8x7b-32768": 0.75,
            "llama-3.1-8b-instant": 0.6
        }

        model_score = model_scores.get(model, 0.5)
        score = (score + model_score) / 2

        # Complexity alignment
        if complexity in [QueryComplexity.SIMPLE, QueryComplexity.MEDIUM]:
            score += 0.1

        return min(1.0, max(0.0, score))

    def _generate_fallback_response(self, query: str, is_follow_up: bool) -> str:
        """Generate a helpful fallback response when AI fails."""

        if is_follow_up:
            return """I apologize, but I'm having trouble processing your follow-up question right now.
            Could you please rephrase your question or try asking about a specific aspect of the prediction analysis?
            I'm here to help explain the student performance prediction results."""
        else:
            return """I apologize, but I'm experiencing technical difficulties generating a detailed analysis right now.
            However, I can tell you that the prediction system has analyzed the student data using multiple machine learning models.
            Please try refreshing the page or contact support if this issue persists."""

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a summary of the current session."""

        if session_id not in self.chat_contexts:
            return {"error": "Session not found"}

        context = self.chat_contexts[session_id]
        history = self.conversation_histories.get(session_id, [])

        return {
            "session_id": session_id,
            "total_interactions": context.total_interactions,
            "session_duration": (datetime.now() - context.session_start_time).total_seconds(),
            "expertise_level": context.user_expertise_level.value,
            "preferred_style": context.preferred_style.value,
            "features_of_interest": context.features_of_interest,
            "conversation_length": len(history),
            "topics_discussed": context.previous_queries[-5:]  # Last 5 topics
        }

    def clear_session(self, session_id: str) -> None:
        """Clear a specific session's data."""

        if session_id in self.conversation_histories:
            del self.conversation_histories[session_id]

        if session_id in self.chat_contexts:
            del self.chat_contexts[session_id]

    def update_user_preferences(
        self,
        session_id: str,
        preferences: Dict[str, Any]
    ) -> bool:
        """Update user preferences for a session."""

        if session_id not in self.chat_contexts:
            self.chat_contexts[session_id] = ChatContext()

        context = self.chat_contexts[session_id]

        try:
            if 'expertise_level' in preferences:
                context.user_expertise_level = ExplanationMode(preferences['expertise_level'])

            if 'response_style' in preferences:
                context.preferred_style = ResponseStyle(preferences['response_style'])

            if 'features_of_interest' in preferences:
                context.features_of_interest = preferences['features_of_interest']

            if 'conversation_focus' in preferences:
                context.conversation_focus = preferences['conversation_focus']

            return True

        except (ValueError, KeyError) as e:
            self.logger.error(f"Error updating preferences: {str(e)}")
            return False


# Utility functions for integration with existing Flask app
def create_enhanced_ai_chat_instance(api_key: str) -> Optional[EnhancedAIChat]:
    """Create and return an enhanced AI chat instance."""
    try:
        return EnhancedAIChat(api_key)
    except Exception as e:
        logging.error(f"Failed to create enhanced AI chat instance: {str(e)}")
        return None


def format_prediction_data_for_enhanced_chat(
    features: Dict[str, Any],
    feature_descriptions: Dict[str, str],
    model_predictions: Dict[str, str],
    primary_prediction: str,
    interpretability_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Format prediction data for the enhanced chat system."""

    return {
        'features': features,
        'feature_descriptions': feature_descriptions,
        'model_predictions': model_predictions,
        'primary_prediction': primary_prediction,
        'interpretability_data': interpretability_data or {}
    }

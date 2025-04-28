"""
Interactive UI for CareBot - An Empathetic Medical Chatbot.
"""

import streamlit as st
from src.models.context_analyzer import ContextAnalyzer
from src.models.empathy_classifier import EmpathyClassifier
from src.models.response_generator import ResponseGenerator

# Configure page
st.set_page_config(page_title="CareBot", page_icon="🤖", layout="wide")

# Initialize session state
if "context_analyzer" not in st.session_state:
    st.session_state.context_analyzer = ContextAnalyzer()
if "empathy_classifier" not in st.session_state:
    st.session_state.empathy_classifier = EmpathyClassifier()
if "response_generator" not in st.session_state:
    st.session_state.response_generator = ResponseGenerator()

# Header
st.title("🤖 CareBot")
st.markdown(
    """
    Your empathetic medical companion. Share your health concerns and receive personalized, 
    understanding responses that address both your medical and emotional needs.
"""
)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    # User input
    user_input = st.text_area(
        "Share your health concerns:",
        placeholder="Example: I've been having severe headaches and feeling dizzy lately...",
        height=150,
    )

    if st.button("Get Response"):
        if user_input:
            with st.spinner("CareBot is analyzing your concerns..."):
                # Analyze context
                medical_context, emotional_context = (
                    st.session_state.context_analyzer.analyze_context(user_input)
                )

                # Generate responses with different empathy levels
                best_response = None
                best_confidence = 0.0
                best_level = None

                for empathy_level in ["high", "medium", "low"]:
                    response = st.session_state.response_generator.generate_response(
                        user_input, medical_context, emotional_context, empathy_level
                    )
                    predicted_level, confidence = (
                        st.session_state.empathy_classifier.predict(response)
                    )

                    if confidence > best_confidence:
                        best_response = response
                        best_confidence = confidence
                        best_level = predicted_level

            # Display results
            st.subheader("Analysis Results")

            # Medical Context
            st.markdown("**Medical Context:**")
            symptoms = [e["text"] for e in medical_context.get("symptoms", [])]
            if symptoms:
                st.markdown(f"Identified symptoms: {', '.join(symptoms)}")
            else:
                st.markdown("No specific symptoms identified")

            # Emotional Context
            st.markdown("**Emotional Context:**")
            emotions = emotional_context.get("emotions", [])
            if emotions:
                st.markdown(f"Detected emotions: {', '.join(emotions)}")
            else:
                st.markdown(
                    "Emotional state: " + emotional_context.get("sentiment", "neutral")
                )

            # Best Response
            st.subheader("CareBot's Response")
            st.markdown(
                f"**Empathy Level:** {best_level.capitalize()} (Confidence: {best_confidence:.2f})"
            )
            st.markdown(best_response)
        else:
            st.warning("Please share your health concerns to get a response.")

with col2:
    # System Information
    st.subheader("About CareBot")
    st.markdown(
        """
        **Your AI Health Companion**
        
        CareBot combines medical expertise with emotional understanding to provide:
        - Symptom analysis and identification
        - Emotional state recognition
        - Empathetic response generation
        - Personalized health guidance
        
        CareBot is designed to be your first point of contact for health concerns,
        offering support while encouraging professional medical consultation when needed.
    """
    )

    # Example Queries
    st.subheader("Try These Examples")
    st.markdown(
        """
        Share concerns like:
        - "I've been having severe headaches and feeling dizzy lately"
        - "My chronic back pain is getting worse and I'm feeling hopeless"
        - "I'm concerned about my high blood pressure and diabetes"
    """
    )

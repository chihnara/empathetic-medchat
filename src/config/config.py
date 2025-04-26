"""
Configuration settings for MEDCOD.
"""

from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
MODEL_DIR = BASE_DIR / "saved_models"
DATA_DIR = BASE_DIR / "data"

# Model configurations
MODEL_CONFIG = {
    "dialogue": {
        "model_name": "microsoft/DialoGPT-medium",
        "max_length": 512,
        "temperature": 0.7,
        "top_p": 0.9,
    },
    "empathy": {
        "model_name": "distilbert-base-uncased",
        "max_length": 128,
    },
}

# Emotional context settings
EMOTIONAL_CONTEXT = {
    "keywords": {
        "fear": ["scared", "afraid", "terrified", "worried", "anxious"],
        "sadness": ["sad", "depressed", "hopeless", "overwhelmed"],
        "pain": ["pain", "hurt", "ache", "sore"],
        "stress": ["stressed", "overwhelmed", "pressure", "tension"],
        "anger": ["angry", "frustrated", "annoyed", "irritated"],
    },
    "intensity_thresholds": {
        "high": 3,
        "medium": 1,
    },
}

# Medical advice templates
MEDICAL_ADVICE = {
    "headache": """Based on your description of severe headaches, I recommend:
1. Taking over-the-counter pain relievers like ibuprofen or acetaminophen
2. Applying a cold or warm compress to your head or neck
3. Resting in a quiet, dark room
4. Staying hydrated and maintaining regular sleep patterns

If your headaches persist or worsen, we should consider additional tests to rule out any underlying conditions.""",
    "sleep": """To help improve your sleep, I recommend:
1. Establishing a regular sleep schedule
2. Creating a relaxing bedtime routine
3. Avoiding screens for at least an hour before bed
4. Keeping your bedroom cool, dark, and quiet
5. Limiting caffeine and alcohol intake

We can also discuss other strategies or potential sleep medications if these measures don't help.""",
    "mental_health": """Regarding your mental health concerns:
1. Let's review your current medication dosage and consider adjustments if needed
2. Consider complementing medication with therapy or counseling
3. Practice stress-reduction techniques like deep breathing or meditation
4. Maintain regular exercise and healthy sleep habits
5. Stay connected with your support system

We can work together to find the right combination of treatments for you.""",
    "diabetes": """For managing your blood sugar levels:
1. Monitor your blood glucose more frequently during this period
2. Keep a detailed food and activity log
3. Review and adjust your medication schedule if needed
4. Ensure regular meals and healthy snacks
5. Incorporate daily physical activity

Let's schedule a follow-up to review your blood sugar logs and make any necessary adjustments to your management plan.""",
    "respiratory": """For your respiratory symptoms:
1. Stay well-hydrated and get plenty of rest
2. Use a humidifier to add moisture to the air
3. Try over-the-counter decongestants or antihistamines
4. Avoid irritants like smoke or strong fragrances
5. Monitor your temperature and any changes in symptoms

If symptoms worsen or you develop fever, we should discuss additional treatment options.""",
}

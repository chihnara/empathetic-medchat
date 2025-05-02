# Empathetic Medical Chat Assistant

An AI-powered medical chat assistant that combines medical knowledge with empathetic responses.

## Features

### V1 (Original Implementation)
- Basic medical context analysis
- Three-level empathy classification (low, medium, high)
- Simple response generation
- Basic conversation state management
- Medical knowledge base integration

### V2 (Enhanced Implementation)
- Fine-grained emotion detection with 8 emotional states
- Enhanced empathy classification with confidence scoring
- Improved response paraphrasing with modern templates
- Comprehensive medical knowledge base integration
- Medical term validation against standard databases
- Enhanced conversation context tracking
- Improved UI with context summary display

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/empathetic-medchat.git
cd empathetic-medchat
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

1. Start the server:
```bash
python app.py
```

2. Access the interfaces:
- V1 (Original): http://localhost:5000/v1
- V2 (Enhanced): http://localhost:5000/v2

## API Endpoints

### V1 Endpoints

- `POST /v1/chat`
  - Input: `{"message": "user message", "session_id": "optional_session_id"}`
  - Output: Basic response with medical and emotional context

- `POST /v1/reset`
  - Input: `{"session_id": "optional_session_id"}`
  - Output: Resets conversation state

### V2 Endpoints

- `POST /v2/chat`
  - Input: `{"message": "user message", "session_id": "optional_session_id"}`
  - Output: Enhanced response with detailed context and confidence scores

- `POST /v2/reset`
  - Input: `{"session_id": "optional_session_id"}`
  - Output: Resets enhanced conversation state

## Enhanced Features (V2)

### 1. Fine-grained Emotion Detection
- Expanded emotion categories:
  - Affirmative
  - Empathy
  - Apology
  - Concern
  - Encouragement
  - Reassurance
  - Acknowledgment
  - None

### 2. Enhanced Empathy Classification
- Multi-level classification with confidence scoring
- Context-aware empathy adjustment
- Improved emotional response templates

### 3. Medical Knowledge Integration
- Comprehensive medical term validation
- Integration with standard medical databases
- Related conditions and symptoms lookup
- Treatment and medication validation

### 4. Improved Response Generation
- Context-aware response templates
- Enhanced paraphrasing capabilities
- Medical accuracy validation
- Empathy-level appropriate responses

### 5. Enhanced UI Features
- Real-time context summary display
- Confidence score visualization
- Medical term validation indicators
- Conversation history tracking

## Project Structure

```
empathetic-medchat/
├── src/
│   ├── models/
│   │   ├── v1/
│   │   │   ├── context_analyzer.py
│   │   │   └── empathy_classifier.py
│   │   └── v2/
│   │       ├── enhanced_empathy_classifier.py
│   │       ├── enhanced_paraphraser.py
│   │       └── medical_knowledge_base.py
│   └── routes/
│       ├── v1_routes.py
│       └── v2_routes.py
├── templates/
│   ├── index.html
│   └── chat_v2.html
├── data/
│   └── medical/
│       ├── conditions.json
│       ├── symptoms.json
│       ├── treatments.json
│       └── medications.json
├── app.py
├── requirements.txt
└── README.md
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the MEDICOD paper implementation
- Uses Hugging Face Transformers
- Medical knowledge from standard medical databases 
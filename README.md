# Empathetic Medical Chat Assistant

An AI-powered medical chat assistant that provides empathetic responses while analyzing medical and emotional context. The system is implemented in two versions (v1 and v2), each with different capabilities and approaches.

## Overview

This project implements an empathetic medical chat assistant that:
- Analyzes medical symptoms and conditions
- Detects emotional context in user messages
- Provides empathetic responses based on both medical and emotional context
- Maintains conversation history and context
- Offers real-time analysis of the conversation

## Versions

### Version 1 (v1)
The base implementation focuses on:
- Basic medical context analysis using biomedical NER
- Simple emotion classification (positive, negative, neutral)
- Three-level empathy classification (low, medium, high)
- Basic conversation state management

### Version 2 (v2)
Enhanced implementation with:
- Fine-grained emotion detection (distress, anxiety, concern, etc.)
- Enhanced empathy classification with confidence scoring
- Improved medical term validation
- Advanced response paraphrasing
- Comprehensive medical knowledge base integration

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/empathetic-medchat.git
cd empathetic-medchat
```

2. Create and activate a virtual environment:
```bash
python -m venv empathetic-medchat-env
source empathetic-medchat-env/bin/activate  # On Windows: .\empathetic-medchat-env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the server:
```bash
python app.py
```

2. Access the interfaces:
- V1 Interface: http://localhost:5000/v1
- V2 Interface: http://localhost:5000/v2 (default)

## Features

### Common Features (v1 & v2)
- Real-time medical context analysis
- Emotional context detection
- Empathetic response generation
- Conversation history tracking
- Context-aware responses
- Interactive web interface

### V2-Specific Enhancements
- More granular emotion detection
- Better confidence scoring for detected emotions
- Enhanced medical term validation
- Improved response paraphrasing
- Integration with medical knowledge base
- More detailed analysis display

## Project Structure

```
empathetic-medchat/
├── src/
│   ├── models/
│   │   ├── v1/
│   │   │   ├── context_analyzer.py     # V1 medical & emotional context analysis
│   │   │   └── empathy_classifier.py   # V1 empathy classification
│   │   └── v2/
│   │       ├── enhanced_empathy_classifier.py   # V2 enhanced emotion analysis
│   │       ├── enhanced_paraphraser.py         # V2 response paraphrasing
│   │       └── medical_knowledge_base.py       # V2 medical term validation
│   ├── routes/
│   │   ├── v1_routes.py   # V1 API endpoints
│   │   └── v2_routes.py   # V2 API endpoints
│   └── utils/
│       └── conversation_state.py   # Shared conversation management
├── templates/
│   ├── common/
│   │   └── base.html      # Base template
│   ├── v1/
│   │   └── chat.html      # V1 interface
│   └── v2/
│       └── chat.html      # V2 interface
├── app.py                 # Main application
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## API Endpoints

### V1 Endpoints
- `GET /v1/` - V1 chat interface
- `POST /v1/chat` - Process messages in V1
- `POST /v1/reset` - Reset V1 conversation

### V2 Endpoints
- `GET /v2/` - V2 chat interface
- `POST /v2/chat` - Process messages in V2
- `POST /v2/reset` - Reset V2 conversation

## Response Format

Both versions return responses in the following format:
```json
{
    "response": "Assistant's response text",
    "medical_context": {
        "symptoms": [{"text": "symptom", "confidence": 0.9}],
        "conditions": [{"text": "condition", "confidence": 0.8}],
        "treatments": [{"text": "treatment", "confidence": 0.7}],
        "medications": [{"text": "medication", "confidence": 0.9}]
    },
    "emotional_context": {
        "emotions": ["emotion1", "emotion2"],
        "empathy_level": "high/medium/low",
        "confidence": 0.85
    },
    "conversation_context": "Summary of current context"
}
```

## Development

The project uses:
- Flask for the web server
- Transformers (Hugging Face) for NLP tasks
- TailwindCSS for styling
- Vanilla JavaScript for frontend interactions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

[Your License Here]

## Acknowledgments

- [MEDCOD paper](https://arxiv.org/abs/2111.09381) (Compton et al., 2021) - *MEDCOD: A Medically-Accurate, Emotive, Diverse, and Controllable Dialog System*
- Hugging Face Transformers library
- Biomedical NER models

## Citation

If you use this project, please cite the MEDCOD paper that inspired it:

```bibtex
@article{compton2021medcod,
  title={MEDCOD: A Medically-Accurate, Emotive, Diverse, and Controllable Dialog System},
  author={Compton, Rhys and Valmianski, Ilya and Deng, Li and Huang, Costa and Katariya, Namit and Amatriain, Xavier and Kannan, Anitha},
  journal={arXiv preprint arXiv:2111.09381},
  year={2021}
}
``` 
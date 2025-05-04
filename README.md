# Empathetic Medical Chat Assistant

An AI-powered medical chat assistant that provides empathetic responses while analyzing medical and emotional context. The system is implemented in two versions (v1 and v2), each with different capabilities and approaches.

## Overview

This project implements an empathetic medical chat assistant that:
- Analyzes medical symptoms and conditions
- Detects emotional context in user messages
- Provides empathetic responses based on both medical and emotional context
- Maintains conversation history and context
- Offers real-time analysis of the conversation
- Integrates an open-source medical knowledge base (v2)
- Supports evaluation and training scripts for research and development

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
- Improved medical term validation and knowledge base integration
- Advanced response paraphrasing
- Comprehensive medical knowledge base integration
- More detailed analysis display
- Improved conversation state management

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

> **Note:**
> - The project uses Hugging Face Transformers, PyTorch, and other NLP libraries. See `requirements.txt` for the full list.
> - For biomedical NER, the model files are expected in `src/models/medical_ner/`.
> - For v2, the medical knowledge base expects data in `data/medical/` (see `src/models/v2/medical_knowledge_base.py`).

## Usage

1. Start the server:
```bash
python app.py
```

2. Access the interfaces:
- V1 Interface: http://localhost:5000/v1
- V2 Interface: http://localhost:5000/v2 (default)
- Root (http://localhost:5000/) redirects to V2

## Features

### Common Features (v1 & v2)
- Real-time medical context analysis (biomedical NER)
- Emotional context detection
- Empathetic response generation
- Conversation history tracking
- Context-aware responses
- Interactive web interface (Flask + TailwindCSS)

### V2-Specific Enhancements
- Fine-grained emotion detection (distress, anxiety, concern, etc.)
- Enhanced empathy classification with confidence scoring
- Medical term validation and normalization using an open-source knowledge base
- Advanced response paraphrasing with LLM integration
- Integration with a medical knowledge base (see `src/models/v2/medical_knowledge_base.py`)
- More detailed analysis and context display
- Improved conversation state management

### Evaluation & Training
- Evaluation scripts and test cases in `src/evaluation/`
- Training scripts for NER and empathy models in `src/training/`
- Configuration files in `src/config/`

## Project Structure

```
empathetic-medchat/
├── src/
│   ├── models/
│   │   ├── v1/
│   │   │   ├── context_analyzer.py           # V1 medical & emotional context analysis
│   │   │   └── empathy_classifier.py         # V1 empathy classification
│   │   ├── v2/
│   │   │   ├── enhanced_empathy_classifier.py   # V2 enhanced emotion analysis
│   │   │   ├── enhanced_paraphraser.py         # V2 response paraphrasing
│   │   │   └── medical_knowledge_base.py       # V2 medical term validation & knowledge base
│   │   └── medical_ner/                      # Biomedical NER model files
│   ├── routes/
│   │   ├── v1_routes.py   # V1 API endpoints
│   │   └── v2_routes.py   # V2 API endpoints
│   ├── utils/
│   │   └── conversation_state.py   # Shared conversation management
│   ├── evaluation/                 # Evaluation scripts and test cases
│   ├── training/                   # Training scripts for models
│   └── config/                     # Configuration files
├── templates/
│   ├── common/
│   │   └── base.html      # Base template
│   ├── v1/
│   │   └── chat.html      # V1 interface
│   └── v2/
│       └── chat.html      # V2 interface
├── data/
│   ├── medical/           # Medical knowledge base data (JSON files)
│   └── medical_dialogues.json  # Example dialogue data
├── docs/                  # Architecture diagrams, reports, and presentations
├── app.py                 # Main application
├── requirements.txt       # Dependencies
├── README.md              # This file
└── ...
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

> **Note:** All endpoints are implemented as Flask blueprints in `src/routes/`.

## Response Format

Both versions return responses in the following format:
```json
{
    "response": "Assistant's response text",
    "medical_context": {
        "symptoms": [{"text": "symptom", "confidence": 0.9}],
        "conditions": [{"text": "condition", "confidence": 0.8}],
        "treatments": [{"text": "treatment", "confidence": 0.7}],
        "medications": [{"text": "medication", "confidence": 0.9}],
        "confidence": 0.85
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
- PyTorch for model inference
- TailwindCSS for styling
- Vanilla JavaScript for frontend interactions
- NLTK, spaCy, scikit-learn, pandas for NLP utilities

### Training & Evaluation
- Training scripts for NER and empathy models: `src/training/`
- Evaluation scripts and test cases: `src/evaluation/`
- Test cases: `src/evaluation/test_cases.json`

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
# empathetic-medchat

This project is an implementation of the MEDICOD (Medical Context and Empathy Detection) system, based on the research paper "MEDICOD: A Medical Context and Empathy Detection System for Medical Dialogue" (Curai Research). The implementation focuses on analyzing medical dialogues to detect both medical context and empathy levels in responses.

## Problem Statement

Medical dialogues often require healthcare providers to balance two critical aspects:
1. Accurate understanding and response to medical conditions and symptoms
2. Appropriate emotional support and empathy in their responses

The challenge lies in developing a system that can:
- Accurately identify medical entities (symptoms, conditions, treatments) in patient queries
- Analyze the emotional context of both patient queries and provider responses
- Classify the level of empathy in provider responses
- Generate appropriate responses that balance medical accuracy with empathetic communication

## Implementation Approach

This implementation attempts to reproduce the core functionality of the original MEDICOD paper while making necessary adaptations for practical deployment. The system consists of several key components:

### 1. Context Analysis
- Medical Entity Recognition (NER) using a fine-tuned BERT model
- Emotional context analysis using keyword-based detection
- Integration of both medical and emotional contexts for comprehensive understanding

### 2. Empathy Classification
- Empathy level classification using a fine-tuned DistilBERT model
- Three-level classification: low, medium, and high empathy
- Confidence scoring for classification results

### 3. Response Generation
- Template-based response generation
- Integration of medical context and empathy levels
- Context-aware response formulation

## Implementation Limitations and Adaptations

### Limitations in Exact Reproduction
1. **Model Size**: The original paper used larger models (e.g., BERT-large), while we use DistilBERT for efficiency
2. **Training Data**: Limited access to the exact training data used in the paper
3. **Computational Resources**: Original implementation used more extensive computational resources
4. **Evaluation Metrics**: Some original evaluation metrics were not reproducible due to data limitations

### Key Adaptations
1. **Model Architecture**:
   - Used DistilBERT instead of BERT for faster inference
   - Simplified the NER pipeline for better performance
   - Implemented caching for frequently seen texts

2. **Training Approach**:
   - Used transfer learning from pre-trained models
   - Implemented smaller batch sizes for memory efficiency
   - Added data augmentation techniques to compensate for limited training data

3. **Evaluation**:
   - Focused on core metrics that could be reliably measured
   - Added confidence scoring for better interpretability
   - Implemented more extensive testing with diverse medical scenarios

## Project Structure

```
empathetic-medchat/
├── app.py                # Flask web server for interactive chat
├── run.py                # Command-line interface for testing
├── requirements.txt      # Project dependencies
├── templates/            # HTML templates for web interface
│   └── index.html        # Main chat interface
├── src/                  # Source code directory
│   ├── models/           # Model implementations
│   │   ├── context_analyzer.py    # Medical and emotional context analysis
│   │   ├── empathy_classifier.py  # Empathy level classification
│   │   ├── response_generator.py  # Response generation
│   │   ├── dialogue_model.py      # Dialogue management
│   │   └── medical_ner/           # Trained medical NER model
│   ├── training/         # Training scripts
│   │   ├── train_ner.py           # NER model training
│   │   ├── prepare_data.py        # Data preparation
│   │   ├── generate_dataset.py    # Dataset generation
│   │   └── test_model.py          # Model testing
│   └── utils/            # Utility functions
├── data/                 # Training and test data
│   └── medical_dialogues.json     # Training data
└── results/              # Output and evaluation results
```

## Key Components

### Context Analyzer
- Analyzes medical entities in text using NER
- Detects emotional context through keyword analysis
- Combines medical and emotional contexts for comprehensive understanding

### Empathy Classifier
- Classifies responses into empathy levels
- Uses a fine-tuned DistilBERT model
- Provides confidence scores for classifications

### Response Generator
- Generates context-aware responses
- Integrates medical and emotional context
- Maintains conversation flow

### Web Interface
- Real-time chat with the medical assistant
- Detailed analysis of each response:
  - Medical context (symptoms, conditions, treatments)
  - Emotional context
  - Empathy level and confidence
  - Conversation context
- Basic session tracking (sessions are not persistent)
- Reset functionality for new conversations
- Modern, responsive design
- Loading indicator while waiting for responses
- Error handling and recovery

## Installation and Usage

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Flask 2.0+
- spaCy 3.0+
- Other dependencies listed in requirements.txt

### Installation
1. Clone the repository
2. Create and activate a virtual environment:
   ```bash
   python -m venv empathetic-medchat-env
   source empathetic-medchat-env/bin/activate  # On Windows: empathetic-medchat-env\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Install spaCy language model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

### Running the System

#### Web Interface (Recommended)
1. Start the Flask server:
   ```bash
   python app.py
   ```
2. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```
3. Start chatting with the medical assistant

#### Command Line Interface
1. Run the test cases:
   ```bash
   python run.py
   ```

#### Training the Model
1. Run the training script:
   ```bash
   python src/training/run_training.py
   ```
   This script will:
   - Load and prepare the medical dialogues data
   - Split the data into training and validation sets
   - Train the NER model
   - Save the trained model to models/medical_ner

Note: The training script requires the medical dialogues data to be present in `data/medical_dialogues.json`. Make sure this file exists before running the training.

### Configuration
- Model paths and parameters can be adjusted in the respective configuration files
- Training parameters can be modified in run_training.py
- Web interface settings can be configured in app.py

## Features

### Web Interface
- Real-time chat with the medical assistant
- Detailed analysis of each response:
  - Medical context (symptoms, conditions, treatments)
  - Emotional context
  - Empathy level and confidence
  - Conversation context
- Reset functionality for new conversations
- Modern, responsive design
- Loading indicator while waiting for responses
- Error handling and recovery

### Command Line Interface
- Test cases with predefined scenarios
- Detailed analysis output
- Error handling and logging

## Future Improvements
1. Integration of more sophisticated emotional analysis
2. Enhanced response generation capabilities
3. Improved training data and evaluation metrics
4. Real-time processing capabilities
5. Integration with medical knowledge bases
6. User authentication and history
7. Export conversation history
8. Mobile app interface

## Acknowledgments
- Original MEDICOD paper authors
- Hugging Face for the Transformers library
- Curai Research for the initial research and inspiration 
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
e2e/
├── run.py                 # Main entry point for the system
├── requirements.txt       # Project dependencies
├── src/                  # Source code directory
│   ├── models/           # Model implementations
│   │   ├── context_analyzer.py    # Medical and emotional context analysis
│   │   ├── empathy_classifier.py  # Empathy level classification
│   │   └── medical_ner/           # Trained medical NER model
│   ├── training/         # Training scripts
│   │   ├── train_ner.py           # NER model training
│   │   ├── prepare_data.py        # Data preparation
│   │   └── test_model.py          # Model testing
│   └── utils/            # Utility functions
├── data/                 # Training and test data
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

### Training Pipeline
- Prepares and processes medical dialogue data
- Trains the NER model for medical entity detection
- Evaluates model performance on test data

## Installation and Usage

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Other dependencies listed in requirements.txt

### Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the System
1. Basic usage:
   ```bash
   python run.py
   ```

2. Training the models:
   ```bash
   python src/training/run_training.py
   ```

3. Testing the models:
   ```bash
   python src/training/test_model.py
   ```

### Configuration
- Model paths and parameters can be adjusted in the respective configuration files
- Training parameters can be modified in the training scripts
- Test cases can be added or modified in run.py

## Future Improvements
1. Integration of more sophisticated emotional analysis
2. Enhanced response generation capabilities
3. Improved training data and evaluation metrics
4. Real-time processing capabilities
5. Integration with medical knowledge bases


## Acknowledgments
- Original MEDICOD paper authors
- Hugging Face for the Transformers library
- Curai Research for the initial research and inspiration 
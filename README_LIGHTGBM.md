# LightGBM for Sarcasm Detection

This repository contains a LightGBM-based model for detecting sarcasm in headlines or short texts. The model combines TF-IDF features with advanced NLP features specifically designed to capture sarcastic patterns.

## Features

The model extracts several types of features that are potentially useful for sarcasm detection:

1. **TF-IDF Features**: Captures important words and phrases
2. **Syntactic Features**: 
   - Sentence length and structure
   - Punctuation patterns (especially question marks and exclamation points)
   - Part-of-speech distributions (noun, verb, adverb, adjective ratios)
   - Text capitalization patterns

3. **Entity Features**:
   - Named entity types and densities
   - Entity distributions (person, organization, location, etc.)

4. **Dependency Features**:
   - Syntactic dependency patterns
   - Tree complexity metrics

5. **Sarcasm-Specific Features**:
   - Sentiment contrast (positive and negative words co-occurring)
   - Extreme adjective usage
   - Quotation patterns

## Getting Started

### Prerequisites

Install the required packages:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

### Running the Model

To train and evaluate the model:

```bash
python training_scripts/lightgbm_sarcasm.py
```

This will:
1. Load and preprocess the data
2. Extract features 
3. Train the LightGBM model
4. Evaluate on validation and test sets
5. Save the model and predictions

### Hyperparameter Tuning

The script includes functionality for hyperparameter optimization. Uncomment the following line in the `main()` function to enable it:

```python
# best_params = optimize_hyperparameters(X_train_text, y_train_labels, X_val_text, y_val_labels)
```

Note that hyperparameter tuning can take a significant amount of time.

## Model Performance

The model will generate evaluation metrics including:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix (saved as 'confusion_matrix.png')
- Feature Importance Plot (saved as 'feature_importance.png')

## Outputs

The script saves:
1. The trained model (`../models/lightgbm_sarcasm_model.pkl`)
2. Predictions on the test set (`../data/lightgbm_predictions.csv`)

## Comparing with BERT

While BERT and other deep learning models excel at capturing contextual meaning, the LightGBM approach:
- Is faster to train and deploy
- Requires less computational resources
- Provides explicit feature importance analysis
- May capture specific sarcasm indicators through engineered features

For best results, consider ensembling both model types.

## Further Improvements

Potential improvements include:
- Adding more sarcasm-specific lexicons
- Incorporating sentiment analysis features
- Using additional language models for feature extraction
- Creating an ensemble of different model types
- Exploring contextualized embeddings as features 
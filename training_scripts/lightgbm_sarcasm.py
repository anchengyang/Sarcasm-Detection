#!/usr/bin/env python

"""
LightGBM for Sarcasm Detection
This script uses LightGBM with NLP features to detect sarcasm in text headlines.
"""

# Standard library imports
import os
import sys
import string
import pickle
from collections import Counter

# Third-party imports
import numpy as np
import pandas as pd
import spacy
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.sparse import issparse, csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

# For handling class imbalance
from imblearn.over_sampling import SMOTE

# Load spaCy model (medium size English model)
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_md")

class SarcasmFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract features that might be useful for sarcasm detection."""
    def __init__(self):
        self.entity_types = None
        self.dependency_types = None
        
    def fit(self, X, y=None):
        """Collect entity types and dependency labels from training data."""
        entity_types_set = set()
        dependency_types_set = set()
        
        for text in X:
            doc = nlp(text)
            # Collect entity types
            for ent in doc.ents:
                entity_types_set.add(ent.label_)
            
            # Collect dependency types
            for token in doc:
                dependency_types_set.add(token.dep_)
                
        self.entity_types = sorted(entity_types_set)
        self.dependency_types = sorted(dependency_types_set)
        return self
    
    def transform(self, X, y=None):
        """Extract NLP features potentially relevant to sarcasm detection."""
        # Basic syntactic features
        syntactic_features = self._extract_syntactic_features(X)
        
        # Entity features (may indicate topic areas)
        entity_features = self._extract_entity_features(X)
        
        # Dependency features (syntactic relationships)
        dependency_features = self._extract_dependency_features(X)
        
        # Sarcasm-specific features
        sarcasm_features = self._extract_sarcasm_specific_features(X)
        
        # Combine all feature sets
        all_features = np.hstack((
            syntactic_features, 
            entity_features, 
            dependency_features,
            sarcasm_features
        ))
        
        return all_features
    
    def _extract_syntactic_features(self, X):
        """Extract syntactic features that might indicate sarcasm."""
        features = []
        for text in X:
            doc = nlp(text)
            sentence_length = len(doc)
            safe_divisor = max(1, sentence_length)
            
            # Basic syntactic features
            punctuation_count = sum(1 for token in doc if token.text in string.punctuation) / safe_divisor
            question_mark_count = sum(1 for token in doc if token.text == "?") / safe_divisor
            exclamation_count = sum(1 for token in doc if token.text == "!") / safe_divisor
            
            # POS features with normalization
            pos_counts = Counter([token.pos_ for token in doc])
            noun_count = pos_counts.get("NOUN", 0) / safe_divisor
            verb_count = pos_counts.get("VERB", 0) / safe_divisor
            adj_count = pos_counts.get("ADJ", 0) / safe_divisor
            adv_count = pos_counts.get("ADV", 0) / safe_divisor  # Adverbs can be important for sarcasm
            intj_count = pos_counts.get("INTJ", 0) / safe_divisor  # Interjections might indicate emotion
            
            # Text structure features
            avg_token_length = sum(len(token.text) for token in doc) / safe_divisor
            uppercase_ratio = sum(1 for token in doc if token.text.isupper()) / safe_divisor
            
            features.append([
                sentence_length,
                punctuation_count,
                question_mark_count,
                exclamation_count,
                noun_count,
                verb_count,
                adj_count,
                adv_count,
                intj_count,
                avg_token_length,
                uppercase_ratio
            ])
        return np.array(features)
    
    def _extract_entity_features(self, X):
        """Extract named entity features, potentially useful for recognizing contexts of sarcasm."""
        features = []
        entity_types_list = self.entity_types or []
        
        if not entity_types_list:
            return np.zeros((len(X), 1))

        for text in X:
            doc = nlp(text)
            total_entities = len(doc.ents)
            safe_divisor = max(1, total_entities)
            
            # Entity type distributions
            type_counts = {etype: 0 for etype in entity_types_list}
            for ent in doc.ents:
                if ent.label_ in type_counts:
                    type_counts[ent.label_] += 1
            
            # Normalized entity counts
            normalized_counts = [type_counts[etype] / safe_divisor for etype in entity_types_list]
            
            # Entity density (entities per token)
            entity_density = total_entities / max(1, len(doc))
            
            feature_row = [total_entities, entity_density] + normalized_counts
            features.append(feature_row)

        return np.array(features)
    
    def _extract_dependency_features(self, X):
        """Extract dependency parsing features that might indicate sarcastic structures."""
        features = []
        dependency_types_list = self.dependency_types or []
        
        if not dependency_types_list:
            return np.zeros((len(X), 1))
            
        for text in X:
            doc = nlp(text)
            token_count = len(doc)
            safe_divisor = max(1, token_count)
            
            # Dependency type distributions
            dep_counts = {dep: 0 for dep in dependency_types_list}
            for token in doc:
                if token.dep_ in dep_counts:
                    dep_counts[token.dep_] += 1
            
            # Normalized dependency counts
            normalized_deps = [dep_counts[dep] / safe_divisor for dep in dependency_types_list]
            
            # Tree complexity features
            root_count = sum(1 for token in doc if token.dep_ == "ROOT")
            avg_children = sum(len(list(token.children)) for token in doc) / safe_divisor
            
            feature_row = [root_count, avg_children] + normalized_deps
            features.append(feature_row)
            
        return np.array(features)
    
    def _extract_sarcasm_specific_features(self, X):
        """Extract features specifically designed for sarcasm detection."""
        features = []
        
        for text in X:
            doc = nlp(text)
            token_count = len(doc)
            safe_divisor = max(1, token_count)
            
            # Sentiment contrast (might indicate sarcasm)
            positive_words = 0
            negative_words = 0
            
            for token in doc:
                # Very basic sentiment detection
                if token.text.lower() in ['good', 'great', 'excellent', 'amazing', 'wonderful', 'happy', 'love', 'best']:
                    positive_words += 1
                elif token.text.lower() in ['bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'sad', 'poor']:
                    negative_words += 1
            
            # Normalized counts
            positive_ratio = positive_words / safe_divisor
            negative_ratio = negative_words / safe_divisor
            
            # Sentiment contrast (high values might indicate sarcasm)
            sentiment_contrast = positive_words > 0 and negative_words > 0
            
            # Extreme adjectives (often used in sarcasm)
            extreme_adj_count = sum(1 for token in doc if token.pos_ == "ADJ" and
                                    token.text.lower() in ["amazing", "incredible", "unbelievable", 
                                                         "fantastic", "terrible", "horrible", 
                                                         "worst", "best"]) / safe_divisor
            
            # Quotation marks (might indicate quoting someone sarcastically)
            quotation_marks = sum(1 for token in doc if token.text in ['"', "'"]) / safe_divisor
            
            features.append([
                positive_ratio,
                negative_ratio,
                int(sentiment_contrast),
                extreme_adj_count,
                quotation_marks
            ])
            
        return np.array(features)


# Combine TF-IDF features with custom NLP features
combined_features = FeatureUnion([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 3))),
    ('sarcasm_features', SarcasmFeatureExtractor()),
])


def train_model(X_train, y_train, params=None):
    """Train the LightGBM model with the given parameters."""
    print("Extracting features...")
    # Create features
    X_train_features = combined_features.fit_transform(X_train)
    
    # Handle class imbalance if needed
    class_counts = np.bincount(y_train)
    if min(class_counts) / max(class_counts) < 0.5:  # If imbalanced
        print(f"Class distribution before SMOTE: {class_counts}")
        
        # Check if we're dealing with a sparse matrix
        if issparse(X_train_features):
            print("Converting sparse matrix for SMOTE processing...")
            # SMOTE can work with sparse matrices, but let's make sure it's in CSR format
            X_train_features = csr_matrix(X_train_features)
        
        smote = SMOTE(random_state=42)
        X_train_features, y_train = smote.fit_resample(X_train_features, y_train)
        print(f"Class distribution after SMOTE: {np.bincount(y_train)}")
    
    # Default parameters
    if params is None:
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'n_estimators': 100,
            'random_state': 42
        }
    
    print("Training model...")
    
    # Check dimensions to decide
    if issparse(X_train_features) and X_train_features.shape[1] < 10000:
        print("Converting sparse features to dense for training...")
        X_train_features = X_train_features.toarray()
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train_features, y_train)
    
    return model, combined_features


def predict(model, feature_pipeline, X_test):
    """Generate predictions using the trained model."""
    print("Predicting...")
    X_test_features = feature_pipeline.transform(X_test)
    
    # LightGBM works better with dense arrays for prediction
    if issparse(X_test_features):
        print("Converting sparse matrix for prediction...")
        X_test_features = X_test_features.toarray()
    
    predictions = model.predict(X_test_features)
    proba_predictions = model.predict_proba(X_test_features)
    
    return predictions, proba_predictions


def evaluate_model(y_true, y_pred, output_dir=None):
    """
    Evaluate the model performance.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_dir: Directory to save the confusion matrix plot (optional)
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes = ['Not Sarcastic', 'Sarcastic']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save with absolute path if output_dir is provided
    if output_dir:
        import os
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path)
        print(f"Confusion matrix saved as '{cm_path}'")
    else:
        plt.savefig('confusion_matrix.png')
        print("Confusion matrix saved as 'confusion_matrix.png'")


def optimize_hyperparameters(X_train, y_train, X_val, y_val):
    """Find the best hyperparameters using RandomizedSearchCV."""
    print("Extracting features for hyperparameter tuning...")
    X_train_features = combined_features.fit_transform(X_train)
    X_val_features = combined_features.transform(X_val)
    
    # Parameter grid for RandomizedSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'num_leaves': [31, 50, 100, 150],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [-1, 5, 10, 15, 20],
        'feature_fraction': [0.7, 0.8, 0.9, 1.0],
        'bagging_fraction': [0.7, 0.8, 0.9, 1.0],
        'min_child_samples': [10, 20, 30, 50]
    }
    
    # Initialize model
    base_model = lgb.LGBMClassifier(
        objective='binary',
        metric='binary_logloss',
        boosting_type='gbdt',
        random_state=42
    )
    
    # RandomizedSearchCV
    print("Starting hyperparameter tuning...")
    search = RandomizedSearchCV(
        base_model, 
        param_grid, 
        n_iter=20,
        scoring='f1',
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    search.fit(X_train_features, y_train)
    
    # Best parameters
    print("Best parameters found:")
    for param, value in search.best_params_.items():
        print(f"{param}: {value}")
    
    # Evaluate best model on validation set
    best_model = search.best_estimator_
    val_predictions = best_model.predict(X_val_features)
    val_accuracy = accuracy_score(y_val, val_predictions)
    val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(y_val, val_predictions, average='binary')
    
    print("\nValidation metrics with best parameters:")
    print(f"Accuracy: {val_accuracy:.4f}")
    print(f"Precision: {val_precision:.4f}")
    print(f"Recall: {val_recall:.4f}")
    print(f"F1 Score: {val_f1:.4f}")
    
    return search.best_params_


def main():
    """Main function to run the sarcasm detection pipeline."""
    # Set paths using proper absolute path resolution
    import os
    
    # Get the project root directory (sarcasm-detection)
    # First get the training_scripts directory, then go one level up
    SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    
    # Define data and model directories using absolute paths
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
    
    # Make sure model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Define file paths
    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")
    train_label_path = os.path.join(DATA_DIR, "train_labels.csv")
    test_label_path = os.path.join(DATA_DIR, "test_labels.csv")
    
    # Load data
    print("Loading data...")
    print(f"Loading from: {DATA_DIR}")
    
    # Check if data files exist
    for path in [train_path, test_path, train_label_path, test_label_path]:
        if not os.path.exists(path):
            print(f"Error: File not found: {path}")
            return
    
    X_train = pd.read_csv(train_path)
    X_test = pd.read_csv(test_path)
    y_train = pd.read_csv(train_label_path)
    y_test = pd.read_csv(test_label_path)
    
    # Get headline column and is_sarcastic target
    X_train_text = X_train['headline']
    y_train_labels = y_train['is_sarcastic'].values
    X_test_text = X_test['headline']
    y_test_labels = y_test['is_sarcastic'].values
    
    # Split training data to create a validation set
    X_train_text, X_val_text, y_train_labels, y_val_labels = train_test_split(
        X_train_text, y_train_labels, test_size=0.2, random_state=42, stratify=y_train_labels
    )
    
    print(f"Training set size: {len(X_train_text)}")
    print(f"Validation set size: {len(X_val_text)}")
    print(f"Test set size: {len(X_test_text)}")
    
    # Uncomment to run hyperparameter optimization (takes time)
    # best_params = optimize_hyperparameters(X_train_text, y_train_labels, X_val_text, y_val_labels)
    
    # Use predefined parameters or the best from optimization
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'n_estimators': 200,
        'max_depth': 15,
        'min_child_samples': 20,
        'random_state': 42
    }
    
    # Train model
    model, feature_pipeline = train_model(X_train_text, y_train_labels, params)
    
    # Evaluate on validation set
    val_predictions, val_proba = predict(model, feature_pipeline, X_val_text)
    print("\nValidation Results:")
    evaluate_model(y_val_labels, val_predictions, PROJECT_ROOT)
    
    # Evaluate on test set
    test_predictions, test_proba = predict(model, feature_pipeline, X_test_text)
    print("\nTest Results:")
    evaluate_model(y_test_labels, test_predictions, PROJECT_ROOT)
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        # For LightGBM's built-in features (not TF-IDF)
        lgb.plot_importance(model, max_num_features=20)
        plt.title('Feature Importance')
        plt.tight_layout()
        
        # Save plot with absolute path
        fig_path = os.path.join(PROJECT_ROOT, 'feature_importance.png')
        plt.savefig(fig_path)
        print(f"Feature importance plot saved as '{fig_path}'")
    
    # Save model and predictions
    print("Saving model and predictions...")
    import pickle
    
    # Save model with absolute path
    model_path = os.path.join(MODEL_DIR, 'lightgbm_sarcasm_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump((model, feature_pipeline), f)
    
    # Save predictions with absolute path
    pred_path = os.path.join(DATA_DIR, 'lightgbm_predictions.csv')
    results_df = pd.DataFrame({
        'headline': X_test['headline'],
        'actual': y_test_labels,
        'predicted': test_predictions,
        'prob_not_sarcastic': test_proba[:, 0],
        'prob_sarcastic': test_proba[:, 1]
    })
    results_df.to_csv(pred_path, index=False)
    
    print(f"All done! Model saved to {model_path}")
    print(f"Predictions saved to {pred_path}")


if __name__ == "__main__":
    main() 
"""
Attention/Focus Classifier for Zense BCI

Classifies mental states (Focused, Relaxed, Neutral, Drowsy) from EEG features.
Provides both traditional ML (Random Forest) and deep learning options.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import pickle

# Scikit-learn imports
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Deep learning (optional)
try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TF = True
except ImportError:
    HAS_TF = False


# Mental state labels
MENTAL_STATES = ['focused', 'relaxed', 'neutral', 'drowsy']


class AttentionClassifier:
    """
    Classifier for detecting attention/mental states from EEG.
    
    Uses band power features and their ratios for classification.
    Supports Random Forest (default), SVM, and Gradient Boosting.
    """
    
    def __init__(self, model_type: str = 'random_forest', n_classes: int = 4):
        """
        Initialize classifier.
        
        Args:
            model_type: 'random_forest', 'svm', 'gradient_boosting', or 'cnn'
            n_classes: Number of mental state classes
        """
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")
        
        self.model_type = model_type
        self.n_classes = n_classes
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        self.feature_names = None
        
        # Initialize base model
        self._init_model()
    
    def _init_model(self):
        """Initialize the ML model based on type."""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif self.model_type == 'cnn':
            if not HAS_TF:
                raise ImportError("TensorFlow required for CNN. Install with: pip install tensorflow")
            self.model = None  # Built during training
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _build_cnn_model(self, input_shape: Tuple[int, ...]) -> 'keras.Model':
        """Build 1D CNN model for raw signal classification."""
        model = keras.Sequential([
            keras.layers.Conv1D(32, 7, activation='relu', padding='same', 
                               input_shape=input_shape),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling1D(2),
            keras.layers.Dropout(0.2),
            
            keras.layers.Conv1D(64, 5, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling1D(2),
            keras.layers.Dropout(0.2),
            
            keras.layers.Conv1D(128, 3, activation='relu', padding='same'),
            keras.layers.GlobalAveragePooling1D(),
            
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(self.n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_features(self, features_list: List[Dict]) -> np.ndarray:
        """
        Convert list of feature dictionaries to array.
        
        Args:
            features_list: List of feature dictionaries
            
        Returns:
            Feature array of shape (n_samples, n_features)
        """
        if not features_list:
            raise ValueError("Empty features list")
        
        # Get feature names from first sample
        if self.feature_names is None:
            self.feature_names = sorted(features_list[0].keys())
        
        X = []
        for features in features_list:
            row = [features.get(name, 0) for name in self.feature_names]
            X.append(row)
        
        return np.array(X)
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            validation_split: float = 0.2) -> Dict[str, float]:
        """
        Train the classifier.
        
        Args:
            X: Feature array of shape (n_samples, n_features)
            y: Labels (string or int)
            validation_split: Fraction for validation
            
        Returns:
            Dictionary with training metrics
        """
        # Encode labels if strings
        if isinstance(y[0], str):
            y = self.label_encoder.fit_transform(y)
        else:
            self.label_encoder.fit(MENTAL_STATES[:self.n_classes])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        if self.model_type == 'cnn':
            # CNN needs different input shape
            if X_scaled.ndim == 2:
                X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
            
            self.model = self._build_cnn_model(X_scaled.shape[1:])
            
            history = self.model.fit(
                X_scaled, y,
                epochs=50,
                batch_size=32,
                validation_split=validation_split,
                verbose=0
            )
            
            metrics = {
                'train_accuracy': history.history['accuracy'][-1],
                'val_accuracy': history.history['val_accuracy'][-1],
            }
        else:
            # Split for validation
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=validation_split, random_state=42
            )
            
            # Train
            self.model.fit(X_train, y_train)
            
            # Evaluate
            train_acc = self.model.score(X_train, y_train)
            val_acc = self.model.score(X_val, y_val)
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, X_scaled, y, cv=5)
            
            metrics = {
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
            }
        
        self.is_fitted = True
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict mental states.
        
        Args:
            X: Feature array
            
        Returns:
            Predicted labels (strings)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        
        if self.model_type == 'cnn' and X_scaled.ndim == 2:
            X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        
        y_pred = self.model.predict(X_scaled)
        
        if self.model_type == 'cnn':
            y_pred = np.argmax(y_pred, axis=1)
        
        return self.label_encoder.inverse_transform(y_pred)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature array
            
        Returns:
            Probability array of shape (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        
        if self.model_type == 'cnn':
            if X_scaled.ndim == 2:
                X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
            return self.model.predict(X_scaled)
        else:
            return self.model.predict_proba(X_scaled)
    
    def classify_realtime(self, features: Dict[str, float]) -> Tuple[str, float]:
        """
        Classify a single sample (for real-time use).
        
        Args:
            features: Feature dictionary
            
        Returns:
            (predicted_state, confidence)
        """
        X = self.prepare_features([features])
        proba = self.predict_proba(X)[0]
        
        pred_idx = np.argmax(proba)
        confidence = proba[pred_idx]
        
        label = self.label_encoder.inverse_transform([pred_idx])[0]
        return label, confidence
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance (for tree-based models)."""
        if not self.is_fitted or self.feature_names is None:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            return dict(zip(self.feature_names, importance))
        
        return None
    
    def save(self, filepath: str):
        """Save model to file."""
        data = {
            'model_type': self.model_type,
            'n_classes': self.n_classes,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
        }
        
        if self.model_type == 'cnn' and HAS_TF:
            # Save Keras model separately
            model_path = Path(filepath).with_suffix('.keras')
            self.model.save(model_path)
            data['keras_model_path'] = str(model_path)
        else:
            data['model'] = self.model
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Model saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'AttentionClassifier':
        """Load model from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        classifier = cls(
            model_type=data['model_type'],
            n_classes=data['n_classes']
        )
        
        classifier.scaler = data['scaler']
        classifier.label_encoder = data['label_encoder']
        classifier.feature_names = data['feature_names']
        classifier.is_fitted = data['is_fitted']
        
        if data['model_type'] == 'cnn' and 'keras_model_path' in data:
            classifier.model = keras.models.load_model(data['keras_model_path'])
        else:
            classifier.model = data['model']
        
        print(f"Model loaded from: {filepath}")
        return classifier


def create_labeled_dataset(recordings, labels: List[str],
                           feature_extractor) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create labeled dataset from recordings.
    
    Args:
        recordings: List of Recording objects
        labels: List of labels corresponding to each recording
        feature_extractor: FeatureExtractor instance
        
    Returns:
        (X, y) arrays
    """
    X_list = []
    y_list = []
    
    for rec, label in zip(recordings, labels):
        # Stack channels
        data = np.vstack([rec.filtered_ch0, rec.filtered_ch1])
        
        # Extract features
        features = feature_extractor.extract_all(data)
        feature_dict = features.to_dict()
        
        X_list.append(list(feature_dict.values()))
        y_list.append(label)
    
    return np.array(X_list), np.array(y_list)

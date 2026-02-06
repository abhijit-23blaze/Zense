"""
Zense BCI Data Analysis Module

Provides offline analysis tools for EEG recordings collected with Zense v0.1.
Includes data loading, preprocessing, feature extraction, and ML-based classification.
"""

from .data_loader import ZenseDataLoader, load_recording, load_directory
from .preprocessing import Preprocessor, filter_signal, epoch_data
from .features import FeatureExtractor, extract_all_features
from .visualization import plot_recording, plot_bands, plot_spectrogram

__version__ = "0.1.0"
__all__ = [
    "ZenseDataLoader",
    "load_recording",
    "load_directory",
    "Preprocessor",
    "filter_signal",
    "epoch_data",
    "FeatureExtractor",
    "extract_all_features",
    "plot_recording",
    "plot_bands",
    "plot_spectrogram",
]

"""
Preprocessing Module for Zense BCI Recordings

Provides signal processing utilities:
- Filtering (bandpass, notch, high-pass, low-pass)
- Epoching (segmentation into analysis windows)
- Normalization (z-score, min-max)
- Artifact rejection
"""

import numpy as np
from scipy import signal
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class Epoch:
    """Container for a single epoch (segment) of EEG data."""
    data: np.ndarray          # Shape: (channels, samples)
    start_sample: int
    end_sample: int
    start_time: float         # In seconds
    duration: float           # In seconds
    label: Optional[str] = None
    is_artifact: bool = False
    
    @property
    def num_samples(self) -> int:
        return self.data.shape[1]
    
    @property
    def num_channels(self) -> int:
        return self.data.shape[0]


class Preprocessor:
    """
    EEG signal preprocessing pipeline.
    
    Provides configurable filtering, epoching, and artifact rejection.
    """
    
    def __init__(self, sample_rate: int = 256):
        """
        Initialize preprocessor.
        
        Args:
            sample_rate: Sampling rate in Hz (256 for UNO, 512 for R4)
        """
        self.sample_rate = sample_rate
        self._filter_cache = {}
    
    def bandpass_filter(self, data: np.ndarray, low_freq: float, high_freq: float, 
                        order: int = 4) -> np.ndarray:
        """
        Apply bandpass filter to signal.
        
        Args:
            data: Input signal (1D or 2D array)
            low_freq: Low cutoff frequency in Hz
            high_freq: High cutoff frequency in Hz  
            order: Filter order
            
        Returns:
            Filtered signal
        """
        nyq = self.sample_rate / 2
        low = low_freq / nyq
        high = high_freq / nyq
        
        # Clip to valid range
        low = max(0.001, min(low, 0.99))
        high = max(low + 0.001, min(high, 0.99))
        
        sos = signal.butter(order, [low, high], btype='band', output='sos')
        
        if data.ndim == 1:
            return signal.sosfiltfilt(sos, data)
        else:
            # Apply to each channel
            return np.array([signal.sosfiltfilt(sos, ch) for ch in data])
    
    def highpass_filter(self, data: np.ndarray, cutoff: float = 0.5, 
                        order: int = 4) -> np.ndarray:
        """Apply high-pass filter to remove DC drift."""
        nyq = self.sample_rate / 2
        normalized_cutoff = cutoff / nyq
        sos = signal.butter(order, normalized_cutoff, btype='highpass', output='sos')
        
        if data.ndim == 1:
            return signal.sosfiltfilt(sos, data)
        else:
            return np.array([signal.sosfiltfilt(sos, ch) for ch in data])
    
    def lowpass_filter(self, data: np.ndarray, cutoff: float = 45, 
                       order: int = 4) -> np.ndarray:
        """Apply low-pass filter for anti-aliasing."""
        nyq = self.sample_rate / 2
        normalized_cutoff = min(cutoff / nyq, 0.99)
        sos = signal.butter(order, normalized_cutoff, btype='lowpass', output='sos')
        
        if data.ndim == 1:
            return signal.sosfiltfilt(sos, data)
        else:
            return np.array([signal.sosfiltfilt(sos, ch) for ch in data])
    
    def notch_filter(self, data: np.ndarray, freq: float = 50, 
                     quality: float = 30) -> np.ndarray:
        """
        Apply notch filter to remove power line interference.
        
        Args:
            data: Input signal
            freq: Frequency to remove (50 Hz for EU, 60 Hz for US)
            quality: Quality factor (higher = narrower notch)
        """
        b, a = signal.iirnotch(freq, quality, self.sample_rate)
        
        if data.ndim == 1:
            return signal.filtfilt(b, a, data)
        else:
            return np.array([signal.filtfilt(b, a, ch) for ch in data])
    
    def apply_standard_filters(self, data: np.ndarray, 
                               highpass: float = 0.5,
                               lowpass: float = 45,
                               notch: float = 50) -> np.ndarray:
        """
        Apply standard EEG preprocessing filter chain.
        
        Args:
            data: Input signal
            highpass: High-pass cutoff (Hz)
            lowpass: Low-pass cutoff (Hz)
            notch: Notch filter frequency (Hz), set to 0 to skip
            
        Returns:
            Filtered signal
        """
        filtered = self.highpass_filter(data, highpass)
        if notch > 0:
            filtered = self.notch_filter(filtered, notch)
        filtered = self.lowpass_filter(filtered, lowpass)
        return filtered
    
    def normalize(self, data: np.ndarray, method: str = 'zscore') -> np.ndarray:
        """
        Normalize signal.
        
        Args:
            data: Input signal
            method: 'zscore' or 'minmax'
            
        Returns:
            Normalized signal
        """
        if method == 'zscore':
            if data.ndim == 1:
                return (data - np.mean(data)) / (np.std(data) + 1e-8)
            else:
                return np.array([(ch - np.mean(ch)) / (np.std(ch) + 1e-8) for ch in data])
        
        elif method == 'minmax':
            if data.ndim == 1:
                return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
            else:
                return np.array([(ch - np.min(ch)) / (np.max(ch) - np.min(ch) + 1e-8) 
                                for ch in data])
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def create_epochs(self, data: np.ndarray, epoch_length: float = 2.0,
                      overlap: float = 0.5, labels: Optional[List[str]] = None) -> List[Epoch]:
        """
        Segment data into epochs.
        
        Args:
            data: Input data, shape (channels, samples) or (samples,)
            epoch_length: Epoch duration in seconds
            overlap: Overlap ratio (0.0 to 0.99)
            labels: Optional labels for each epoch
            
        Returns:
            List of Epoch objects
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        n_channels, n_samples = data.shape
        samples_per_epoch = int(epoch_length * self.sample_rate)
        step = int(samples_per_epoch * (1 - overlap))
        
        epochs = []
        epoch_idx = 0
        
        for start in range(0, n_samples - samples_per_epoch + 1, step):
            end = start + samples_per_epoch
            epoch_data = data[:, start:end]
            
            label = labels[epoch_idx] if labels and epoch_idx < len(labels) else None
            
            epochs.append(Epoch(
                data=epoch_data,
                start_sample=start,
                end_sample=end,
                start_time=start / self.sample_rate,
                duration=epoch_length,
                label=label
            ))
            epoch_idx += 1
        
        return epochs
    
    def detect_artifacts(self, epochs: List[Epoch], 
                         amplitude_threshold: float = 150,
                         gradient_threshold: float = 50) -> List[Epoch]:
        """
        Mark epochs containing artifacts.
        
        Args:
            epochs: List of epochs to check
            amplitude_threshold: Max absolute amplitude (µV equivalent)
            gradient_threshold: Max sample-to-sample difference
            
        Returns:
            Same epochs with is_artifact flag updated
        """
        for epoch in epochs:
            # Check amplitude
            max_amp = np.max(np.abs(epoch.data))
            if max_amp > amplitude_threshold:
                epoch.is_artifact = True
                continue
            
            # Check gradient (sample-to-sample changes)
            gradient = np.diff(epoch.data, axis=1)
            max_gradient = np.max(np.abs(gradient))
            if max_gradient > gradient_threshold:
                epoch.is_artifact = True
        
        return epochs
    
    def reject_artifacts(self, epochs: List[Epoch]) -> List[Epoch]:
        """Remove epochs marked as artifacts."""
        return [e for e in epochs if not e.is_artifact]
    
    def epochs_to_array(self, epochs: List[Epoch]) -> np.ndarray:
        """
        Convert list of epochs to 3D array.
        
        Returns:
            Array of shape (n_epochs, n_channels, n_samples)
        """
        return np.array([e.data for e in epochs])


# Convenience functions
def filter_signal(data: np.ndarray, sample_rate: int = 256,
                  highpass: float = 0.5, lowpass: float = 45,
                  notch: float = 50) -> np.ndarray:
    """Apply standard EEG filter chain."""
    prep = Preprocessor(sample_rate)
    return prep.apply_standard_filters(data, highpass, lowpass, notch)


def epoch_data(data: np.ndarray, sample_rate: int = 256,
               epoch_length: float = 2.0, overlap: float = 0.5,
               reject_artifacts: bool = True) -> List[Epoch]:
    """Segment data into epochs with optional artifact rejection."""
    prep = Preprocessor(sample_rate)
    epochs = prep.create_epochs(data, epoch_length, overlap)
    if reject_artifacts:
        epochs = prep.detect_artifacts(epochs)
        epochs = prep.reject_artifacts(epochs)
    return epochs

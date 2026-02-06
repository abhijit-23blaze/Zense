"""
Feature Extraction for Zense BCI Recordings

Extracts features from EEG signals in multiple domains:
- Time domain: statistical features, Hjorth parameters
- Frequency domain: band powers, spectral features  
- Wavelet domain: sub-band energies
"""

import numpy as np
from scipy import signal, stats
from scipy.fft import rfft, rfftfreq
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

# Try importing pywt for wavelet features (optional)
try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False


# Standard EEG frequency bands
FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45),
}


@dataclass
class Features:
    """Container for extracted features from an epoch or recording."""
    
    # Time domain features per channel
    time_features: Dict[str, np.ndarray]
    
    # Frequency domain features per channel
    freq_features: Dict[str, np.ndarray]
    
    # Band powers per channel
    band_powers: Dict[str, np.ndarray]
    
    # Wavelet features per channel (optional)
    wavelet_features: Optional[Dict[str, np.ndarray]] = None
    
    # Derived features
    band_ratios: Optional[Dict[str, float]] = None
    
    def to_vector(self) -> np.ndarray:
        """Flatten all features into a single feature vector."""
        vectors = []
        
        for key in sorted(self.time_features.keys()):
            vectors.append(np.atleast_1d(self.time_features[key]).flatten())
        
        for key in sorted(self.freq_features.keys()):
            vectors.append(np.atleast_1d(self.freq_features[key]).flatten())
        
        for key in sorted(self.band_powers.keys()):
            vectors.append(np.atleast_1d(self.band_powers[key]).flatten())
        
        if self.wavelet_features:
            for key in sorted(self.wavelet_features.keys()):
                vectors.append(np.atleast_1d(self.wavelet_features[key]).flatten())
        
        if self.band_ratios:
            for key in sorted(self.band_ratios.keys()):
                vectors.append(np.atleast_1d(self.band_ratios[key]).flatten())
        
        return np.concatenate(vectors)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert all features to a flat dictionary."""
        result = {}
        
        for key, value in self.time_features.items():
            if isinstance(value, np.ndarray):
                for i, v in enumerate(value.flatten()):
                    result[f"time_{key}_ch{i}"] = v
            else:
                result[f"time_{key}"] = value
        
        for key, value in self.freq_features.items():
            if isinstance(value, np.ndarray):
                for i, v in enumerate(value.flatten()):
                    result[f"freq_{key}_ch{i}"] = v
            else:
                result[f"freq_{key}"] = value
        
        for key, value in self.band_powers.items():
            if isinstance(value, np.ndarray):
                for i, v in enumerate(value.flatten()):
                    result[f"band_{key}_ch{i}"] = v
            else:
                result[f"band_{key}"] = value
        
        if self.band_ratios:
            for key, value in self.band_ratios.items():
                result[f"ratio_{key}"] = value
        
        return result


class FeatureExtractor:
    """
    Extract features from EEG signals.
    
    Supports time-domain, frequency-domain, and wavelet-domain features.
    """
    
    def __init__(self, sample_rate: int = 256, 
                 freq_bands: Optional[Dict[str, Tuple[float, float]]] = None):
        """
        Initialize feature extractor.
        
        Args:
            sample_rate: Sampling rate in Hz
            freq_bands: Dictionary of frequency band definitions
        """
        self.sample_rate = sample_rate
        self.freq_bands = freq_bands or FREQ_BANDS
    
    # ==================== Time Domain Features ====================
    
    def extract_time_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract time-domain features.
        
        Args:
            data: Input signal, shape (channels, samples) or (samples,)
            
        Returns:
            Dictionary of feature arrays
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        n_channels = data.shape[0]
        
        features = {
            'mean': np.zeros(n_channels),
            'std': np.zeros(n_channels),
            'var': np.zeros(n_channels),
            'skewness': np.zeros(n_channels),
            'kurtosis': np.zeros(n_channels),
            'rms': np.zeros(n_channels),
            'ptp': np.zeros(n_channels),  # Peak-to-peak
            'zero_crossings': np.zeros(n_channels),
            'hjorth_activity': np.zeros(n_channels),
            'hjorth_mobility': np.zeros(n_channels),
            'hjorth_complexity': np.zeros(n_channels),
        }
        
        for ch in range(n_channels):
            x = data[ch]
            
            # Basic statistics
            features['mean'][ch] = np.mean(x)
            features['std'][ch] = np.std(x)
            features['var'][ch] = np.var(x)
            features['skewness'][ch] = stats.skew(x)
            features['kurtosis'][ch] = stats.kurtosis(x)
            features['rms'][ch] = np.sqrt(np.mean(x**2))
            features['ptp'][ch] = np.ptp(x)
            
            # Zero crossings
            features['zero_crossings'][ch] = np.sum(np.diff(np.sign(x)) != 0)
            
            # Hjorth parameters
            dx = np.diff(x)
            ddx = np.diff(dx)
            
            var_x = np.var(x)
            var_dx = np.var(dx)
            var_ddx = np.var(ddx)
            
            features['hjorth_activity'][ch] = var_x
            features['hjorth_mobility'][ch] = np.sqrt(var_dx / (var_x + 1e-10))
            features['hjorth_complexity'][ch] = (
                np.sqrt(var_ddx / (var_dx + 1e-10)) / 
                (features['hjorth_mobility'][ch] + 1e-10)
            )
        
        return features
    
    # ==================== Frequency Domain Features ====================
    
    def compute_psd(self, data: np.ndarray, 
                    method: str = 'welch') -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute power spectral density.
        
        Args:
            data: Input signal (1D)
            method: 'welch' or 'fft'
            
        Returns:
            (frequencies, power)
        """
        if method == 'welch':
            nperseg = min(len(data), self.sample_rate * 2)
            freqs, psd = signal.welch(data, fs=self.sample_rate, nperseg=nperseg)
        else:  # FFT
            n = len(data)
            fft_vals = rfft(data * signal.windows.hamming(n))
            psd = np.abs(fft_vals) ** 2
            freqs = rfftfreq(n, 1/self.sample_rate)
        
        return freqs, psd
    
    def compute_band_power(self, freqs: np.ndarray, psd: np.ndarray,
                           band: Tuple[float, float]) -> float:
        """Compute power in a frequency band."""
        idx = np.logical_and(freqs >= band[0], freqs <= band[1])
        return np.sum(psd[idx])
    
    def extract_freq_features(self, data: np.ndarray) -> Tuple[Dict[str, np.ndarray], 
                                                                Dict[str, np.ndarray]]:
        """
        Extract frequency-domain features.
        
        Args:
            data: Input signal, shape (channels, samples) or (samples,)
            
        Returns:
            (spectral_features, band_powers)
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        n_channels = data.shape[0]
        
        freq_features = {
            'spectral_entropy': np.zeros(n_channels),
            'peak_frequency': np.zeros(n_channels),
            'mean_frequency': np.zeros(n_channels),
            'median_frequency': np.zeros(n_channels),
            'spectral_edge_90': np.zeros(n_channels),
        }
        
        band_powers = {band: np.zeros(n_channels) for band in self.freq_bands}
        band_powers['relative'] = {}
        
        for ch in range(n_channels):
            freqs, psd = self.compute_psd(data[ch])
            
            # Normalize PSD for relative measures
            psd_norm = psd / (np.sum(psd) + 1e-10)
            
            # Spectral entropy
            psd_prob = psd_norm + 1e-10
            freq_features['spectral_entropy'][ch] = -np.sum(psd_prob * np.log2(psd_prob))
            
            # Peak frequency
            freq_features['peak_frequency'][ch] = freqs[np.argmax(psd)]
            
            # Mean and median frequency
            freq_features['mean_frequency'][ch] = np.sum(freqs * psd_norm)
            cumsum = np.cumsum(psd_norm)
            freq_features['median_frequency'][ch] = freqs[np.searchsorted(cumsum, 0.5)]
            
            # Spectral edge (90% of power)
            freq_features['spectral_edge_90'][ch] = freqs[np.searchsorted(cumsum, 0.9)]
            
            # Band powers
            total_power = np.sum(psd)
            for band_name, band_range in self.freq_bands.items():
                power = self.compute_band_power(freqs, psd, band_range)
                band_powers[band_name][ch] = power
                
                # Relative power
                if band_name not in band_powers['relative']:
                    band_powers['relative'][band_name] = np.zeros(n_channels)
                band_powers['relative'][band_name][ch] = power / (total_power + 1e-10) * 100
        
        return freq_features, band_powers
    
    def compute_band_ratios(self, band_powers: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute commonly used band power ratios.
        
        Returns ratios averaged across channels.
        """
        ratios = {}
        
        # Alpha/Beta ratio (relaxation indicator)
        alpha = np.mean(band_powers.get('alpha', [0]))
        beta = np.mean(band_powers.get('beta', [1]))
        ratios['alpha_beta'] = alpha / (beta + 1e-10)
        
        # Theta/Beta ratio (attention indicator)
        theta = np.mean(band_powers.get('theta', [0]))
        ratios['theta_beta'] = theta / (beta + 1e-10)
        
        # Alpha/Theta ratio
        ratios['alpha_theta'] = alpha / (theta + 1e-10)
        
        # (Theta + Alpha) / Beta (engagement indicator)
        ratios['engagement'] = (theta + alpha) / (beta + 1e-10)
        
        # Delta/Alpha ratio
        delta = np.mean(band_powers.get('delta', [0]))
        ratios['delta_alpha'] = delta / (alpha + 1e-10)
        
        return ratios
    
    # ==================== Wavelet Features ====================
    
    def extract_wavelet_features(self, data: np.ndarray, 
                                  wavelet: str = 'db4',
                                  level: int = 4) -> Dict[str, np.ndarray]:
        """
        Extract wavelet-domain features using DWT.
        
        Args:
            data: Input signal, shape (channels, samples) or (samples,)
            wavelet: Wavelet type (e.g., 'db4', 'sym5')
            level: Decomposition level
            
        Returns:
            Dictionary of wavelet features
        """
        if not HAS_PYWT:
            return {}
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        n_channels = data.shape[0]
        
        features = {
            'wavelet_energy': np.zeros((n_channels, level + 1)),
            'wavelet_entropy': np.zeros((n_channels, level + 1)),
        }
        
        for ch in range(n_channels):
            # Perform DWT
            coeffs = pywt.wavedec(data[ch], wavelet, level=level)
            
            for i, coeff in enumerate(coeffs):
                # Energy
                energy = np.sum(coeff ** 2)
                features['wavelet_energy'][ch, i] = energy
                
                # Entropy
                coeff_norm = np.abs(coeff) / (np.sum(np.abs(coeff)) + 1e-10)
                entropy = -np.sum(coeff_norm * np.log2(coeff_norm + 1e-10))
                features['wavelet_entropy'][ch, i] = entropy
        
        return features
    
    # ==================== Main Extraction Method ====================
    
    def extract_all(self, data: np.ndarray, 
                    include_wavelet: bool = True) -> Features:
        """
        Extract all features from data.
        
        Args:
            data: Input signal, shape (channels, samples) or (samples,)
            include_wavelet: Whether to include wavelet features
            
        Returns:
            Features object containing all extracted features
        """
        time_features = self.extract_time_features(data)
        freq_features, band_powers = self.extract_freq_features(data)
        band_ratios = self.compute_band_ratios(band_powers)
        
        wavelet_features = None
        if include_wavelet and HAS_PYWT:
            wavelet_features = self.extract_wavelet_features(data)
        
        return Features(
            time_features=time_features,
            freq_features=freq_features,
            band_powers=band_powers,
            wavelet_features=wavelet_features,
            band_ratios=band_ratios,
        )


# Convenience function
def extract_all_features(data: np.ndarray, sample_rate: int = 256,
                         include_wavelet: bool = True) -> Features:
    """Extract all features from EEG data."""
    extractor = FeatureExtractor(sample_rate)
    return extractor.extract_all(data, include_wavelet)

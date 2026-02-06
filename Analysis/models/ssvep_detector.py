"""
SSVEP Detector for Zense BCI

Detects Steady-State Visual Evoked Potentials (SSVEP) at specific frequencies.
Uses Canonical Correlation Analysis (CCA) and FFT-based methods.
"""

import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class SSVEPResult:
    """Result of SSVEP detection."""
    detected_frequency: Optional[float]
    confidence: float
    all_correlations: Dict[float, float]
    snr: float
    is_detected: bool


class SSVEPDetector:
    """
    Detector for SSVEP responses.
    
    Supports multiple detection methods:
    - CCA (Canonical Correlation Analysis)
    - FFT Peak Detection
    - Power Spectral Density matching
    """
    
    def __init__(self, sample_rate: int = 256, 
                 target_frequencies: Optional[List[float]] = None,
                 n_harmonics: int = 2):
        """
        Initialize SSVEP detector.
        
        Args:
            sample_rate: Sampling rate in Hz
            target_frequencies: List of SSVEP frequencies to detect (Hz)
            n_harmonics: Number of harmonics to include in reference signals
        """
        self.sample_rate = sample_rate
        self.target_frequencies = target_frequencies or [10, 12, 15]  # Common SSVEP frequencies
        self.n_harmonics = n_harmonics
        
        # Precompute reference signals
        self._reference_signals = {}
    
    def generate_reference(self, frequency: float, n_samples: int) -> np.ndarray:
        """
        Generate reference signals for a target frequency.
        
        Creates sin and cos at the fundamental frequency and harmonics.
        
        Args:
            frequency: Target frequency in Hz
            n_samples: Number of samples
            
        Returns:
            Reference matrix of shape (2 * n_harmonics, n_samples)
        """
        t = np.arange(n_samples) / self.sample_rate
        refs = []
        
        for h in range(1, self.n_harmonics + 1):
            f = frequency * h
            refs.append(np.sin(2 * np.pi * f * t))
            refs.append(np.cos(2 * np.pi * f * t))
        
        return np.array(refs)
    
    def cca(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute Canonical Correlation Analysis between EEG and reference.
        
        Args:
            X: EEG data of shape (channels, samples) or (samples,)
            Y: Reference signals of shape (refs, samples)
            
        Returns:
            Maximum canonical correlation
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Center the data
        X = X - X.mean(axis=1, keepdims=True)
        Y = Y - Y.mean(axis=1, keepdims=True)
        
        n_samples = X.shape[1]
        
        # Compute covariance matrices
        Cxx = X @ X.T / n_samples
        Cyy = Y @ Y.T / n_samples
        Cxy = X @ Y.T / n_samples
        Cyx = Y @ X.T / n_samples
        
        # Regularization
        reg = 1e-6
        Cxx += reg * np.eye(Cxx.shape[0])
        Cyy += reg * np.eye(Cyy.shape[0])
        
        # Solve generalized eigenvalue problem
        try:
            Cxx_inv = np.linalg.inv(Cxx)
            Cyy_inv = np.linalg.inv(Cyy)
            
            M = Cxx_inv @ Cxy @ Cyy_inv @ Cyx
            eigenvalues = np.linalg.eigvals(M)
            
            # Return maximum correlation
            max_corr = np.sqrt(np.max(np.abs(eigenvalues.real)))
            return min(max_corr, 1.0)  # Clip to valid range
            
        except np.linalg.LinAlgError:
            return 0.0
    
    def detect_fft(self, data: np.ndarray) -> Tuple[float, Dict[float, float]]:
        """
        Detect SSVEP using FFT peak detection.
        
        Args:
            data: EEG signal (1D or 2D)
            
        Returns:
            (detected_frequency, {freq: power} dict)
        """
        if data.ndim == 2:
            # Average across channels
            data = np.mean(data, axis=0)
        
        # Compute FFT
        n = len(data)
        windowed = data * signal.windows.hamming(n)
        fft_vals = rfft(windowed)
        power = np.abs(fft_vals) ** 2
        freqs = rfftfreq(n, 1/self.sample_rate)
        
        # Check power at target frequencies
        freq_powers = {}
        for target_freq in self.target_frequencies:
            # Find closest frequency bin
            idx = np.argmin(np.abs(freqs - target_freq))
            
            # Sum power in small window around target
            window = 2  # +/- 2 bins
            start = max(0, idx - window)
            end = min(len(power), idx + window + 1)
            freq_powers[target_freq] = np.sum(power[start:end])
        
        # Find frequency with highest power
        if freq_powers:
            detected = max(freq_powers, key=freq_powers.get)
            return detected, freq_powers
        
        return None, {}
    
    def compute_snr(self, data: np.ndarray, frequency: float) -> float:
        """
        Compute SNR at a target frequency.
        
        Args:
            data: EEG signal
            frequency: Target frequency
            
        Returns:
            SNR in dB
        """
        if data.ndim == 2:
            data = np.mean(data, axis=0)
        
        # Compute PSD
        nperseg = min(len(data), self.sample_rate * 2)
        freqs, psd = signal.welch(data, fs=self.sample_rate, nperseg=nperseg)
        
        # Find target frequency bin
        idx = np.argmin(np.abs(freqs - frequency))
        
        # Signal power: target frequency and harmonics
        signal_power = psd[idx]
        for h in range(2, self.n_harmonics + 1):
            harm_idx = np.argmin(np.abs(freqs - frequency * h))
            if harm_idx < len(psd):
                signal_power += psd[harm_idx]
        
        # Noise power: average of surrounding frequencies (excluding signal)
        noise_mask = np.ones(len(psd), dtype=bool)
        for freq in [frequency * h for h in range(1, self.n_harmonics + 1)]:
            freq_idx = np.argmin(np.abs(freqs - freq))
            noise_mask[max(0, freq_idx-2):min(len(psd), freq_idx+3)] = False
        
        noise_power = np.mean(psd[noise_mask]) if np.any(noise_mask) else 1e-10
        
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        return snr
    
    def detect(self, data: np.ndarray, 
               method: str = 'cca',
               threshold: float = 0.3) -> SSVEPResult:
        """
        Detect SSVEP in EEG data.
        
        Args:
            data: EEG data of shape (channels, samples) or (samples,)
            method: 'cca' or 'fft'
            threshold: Detection threshold (correlation for CCA, relative power for FFT)
            
        Returns:
            SSVEPResult with detection outcome
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        n_samples = data.shape[1]
        correlations = {}
        
        if method == 'cca':
            # CCA-based detection
            for freq in self.target_frequencies:
                ref = self.generate_reference(freq, n_samples)
                corr = self.cca(data, ref)
                correlations[freq] = corr
            
            # Find best frequency
            if correlations:
                best_freq = max(correlations, key=correlations.get)
                best_corr = correlations[best_freq]
                
                is_detected = best_corr >= threshold
                detected_freq = best_freq if is_detected else None
                snr = self.compute_snr(data, best_freq) if is_detected else 0.0
                
                return SSVEPResult(
                    detected_frequency=detected_freq,
                    confidence=best_corr,
                    all_correlations=correlations,
                    snr=snr,
                    is_detected=is_detected
                )
        
        else:  # FFT method
            detected_freq, freq_powers = self.detect_fft(data)
            
            # Normalize powers
            total = sum(freq_powers.values()) + 1e-10
            correlations = {f: p/total for f, p in freq_powers.items()}
            
            if detected_freq is not None:
                confidence = correlations[detected_freq]
                is_detected = confidence >= threshold
                snr = self.compute_snr(data, detected_freq)
                
                return SSVEPResult(
                    detected_frequency=detected_freq if is_detected else None,
                    confidence=confidence,
                    all_correlations=correlations,
                    snr=snr,
                    is_detected=is_detected
                )
        
        return SSVEPResult(
            detected_frequency=None,
            confidence=0.0,
            all_correlations=correlations,
            snr=0.0,
            is_detected=False
        )
    
    def analyze_recording(self, data: np.ndarray, 
                          epoch_length: float = 4.0,
                          overlap: float = 0.5) -> List[SSVEPResult]:
        """
        Analyze a recording by detecting SSVEP in epochs.
        
        Args:
            data: EEG data
            epoch_length: Length of each epoch in seconds
            overlap: Overlap ratio between epochs
            
        Returns:
            List of SSVEPResult for each epoch
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        n_samples = data.shape[1]
        samples_per_epoch = int(epoch_length * self.sample_rate)
        step = int(samples_per_epoch * (1 - overlap))
        
        results = []
        for start in range(0, n_samples - samples_per_epoch + 1, step):
            end = start + samples_per_epoch
            epoch = data[:, start:end]
            
            result = self.detect(epoch)
            results.append(result)
        
        return results
    
    def get_detection_summary(self, results: List[SSVEPResult]) -> Dict:
        """
        Summarize detection results across epochs.
        
        Args:
            results: List of SSVEPResult
            
        Returns:
            Summary dictionary
        """
        detected = [r for r in results if r.is_detected]
        
        freq_counts = {}
        for r in detected:
            freq = r.detected_frequency
            freq_counts[freq] = freq_counts.get(freq, 0) + 1
        
        return {
            'total_epochs': len(results),
            'detected_epochs': len(detected),
            'detection_rate': len(detected) / len(results) if results else 0,
            'frequency_counts': freq_counts,
            'dominant_frequency': max(freq_counts, key=freq_counts.get) if freq_counts else None,
            'mean_confidence': np.mean([r.confidence for r in detected]) if detected else 0,
            'mean_snr': np.mean([r.snr for r in detected]) if detected else 0,
        }

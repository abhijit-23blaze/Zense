"""
Artifact Detector for Zense BCI

Automatically detects and marks EEG artifacts:
- Eye blinks
- Muscle artifacts
- Electrode pops/disconnections
- Motion artifacts
"""

import numpy as np
from scipy import signal, stats
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class ArtifactEvent:
    """Represents a detected artifact."""
    start_sample: int
    end_sample: int
    start_time: float      # Seconds
    end_time: float        # Seconds
    artifact_type: str     # 'blink', 'muscle', 'electrode', 'motion', 'unknown'
    severity: float        # 0-1 scale
    channel: Optional[int] = None
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class ArtifactDetector:
    """
    Detect artifacts in EEG data.
    
    Methods:
    - Amplitude thresholding
    - Gradient (derivative) analysis
    - Statistical outlier detection
    - Frequency-based detection
    """
    
    def __init__(self, sample_rate: int = 256):
        """
        Initialize artifact detector.
        
        Args:
            sample_rate: Sampling rate in Hz
        """
        self.sample_rate = sample_rate
        
        # Default thresholds (can be calibrated)
        self.amplitude_threshold = 150    # Max amplitude for clean EEG
        self.gradient_threshold = 50      # Max sample-to-sample change
        self.zscore_threshold = 4.0       # Z-score for outlier detection
        self.min_artifact_duration = 0.05 # Minimum artifact duration (50ms)
    
    def set_thresholds(self, amplitude: float = None, gradient: float = None,
                       zscore: float = None):
        """Update detection thresholds."""
        if amplitude is not None:
            self.amplitude_threshold = amplitude
        if gradient is not None:
            self.gradient_threshold = gradient
        if zscore is not None:
            self.zscore_threshold = zscore
    
    def calibrate_from_data(self, clean_data: np.ndarray, 
                            multiplier: float = 3.0):
        """
        Calibrate thresholds from known clean data.
        
        Args:
            clean_data: Known artifact-free EEG segment
            multiplier: Threshold = multiplier * std
        """
        if clean_data.ndim == 2:
            clean_data = clean_data.flatten()
        
        self.amplitude_threshold = multiplier * np.std(clean_data) + np.mean(np.abs(clean_data))
        self.gradient_threshold = multiplier * np.std(np.diff(clean_data))
        
        print(f"Calibrated thresholds: amplitude={self.amplitude_threshold:.1f}, "
              f"gradient={self.gradient_threshold:.1f}")
    
    def detect_amplitude_artifacts(self, data: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect artifacts based on amplitude threshold.
        
        Args:
            data: 1D signal
            
        Returns:
            List of (start, end) sample indices for artifact regions
        """
        # Find samples exceeding threshold
        artifact_mask = np.abs(data) > self.amplitude_threshold
        
        return self._mask_to_regions(artifact_mask)
    
    def detect_gradient_artifacts(self, data: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect artifacts based on gradient (sudden jumps).
        
        Args:
            data: 1D signal
            
        Returns:
            List of (start, end) sample indices
        """
        gradient = np.abs(np.diff(data))
        artifact_mask = np.zeros(len(data), dtype=bool)
        artifact_mask[:-1] = gradient > self.gradient_threshold
        artifact_mask[1:] |= gradient > self.gradient_threshold
        
        return self._mask_to_regions(artifact_mask)
    
    def detect_statistical_outliers(self, data: np.ndarray, 
                                     window_size: float = 1.0) -> List[Tuple[int, int]]:
        """
        Detect artifacts using rolling Z-score.
        
        Args:
            data: 1D signal
            window_size: Window size in seconds for local statistics
            
        Returns:
            List of (start, end) sample indices
        """
        window = int(window_size * self.sample_rate)
        n = len(data)
        
        artifact_mask = np.zeros(n, dtype=bool)
        
        for i in range(0, n - window, window // 2):
            segment = data[i:i+window]
            z_scores = np.abs(stats.zscore(segment))
            outliers = z_scores > self.zscore_threshold
            artifact_mask[i:i+window] |= outliers
        
        return self._mask_to_regions(artifact_mask)
    
    def detect_muscle_artifacts(self, data: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect muscle artifacts (high-frequency bursts).
        
        Muscle artifacts typically have high energy in beta/gamma bands.
        
        Args:
            data: 1D signal
            
        Returns:
            List of (start, end) sample indices
        """
        # High-pass filter to isolate high-frequency content
        nyq = self.sample_rate / 2
        high_freq = min(20 / nyq, 0.95)
        sos = signal.butter(2, high_freq, btype='highpass', output='sos')
        high_passed = signal.sosfiltfilt(sos, data)
        
        # Compute envelope using Hilbert transform
        analytic = signal.hilbert(high_passed)
        envelope = np.abs(analytic)
        
        # Smooth envelope
        window = int(0.1 * self.sample_rate)  # 100ms window
        envelope_smooth = np.convolve(envelope, np.ones(window)/window, mode='same')
        
        # Threshold based on statistics
        threshold = np.mean(envelope_smooth) + 2 * np.std(envelope_smooth)
        artifact_mask = envelope_smooth > threshold
        
        return self._mask_to_regions(artifact_mask)
    
    def detect_blink_artifacts(self, data: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect eye blink artifacts.
        
        Blinks are characterized by slow, large-amplitude deflections.
        
        Args:
            data: 1D signal
            
        Returns:
            List of (start, end) sample indices
        """
        # Low-pass filter to isolate blink frequency (0.5-5 Hz)
        nyq = self.sample_rate / 2
        low_freq = min(5 / nyq, 0.95)
        sos = signal.butter(2, low_freq, btype='lowpass', output='sos')
        low_passed = signal.sosfiltfilt(sos, data)
        
        # Look for large deflections
        threshold = 2 * np.std(low_passed)
        artifact_mask = np.abs(low_passed) > threshold
        
        # Blinks are typically 100-400ms
        regions = self._mask_to_regions(artifact_mask)
        
        # Filter by duration
        blink_regions = []
        for start, end in regions:
            duration = (end - start) / self.sample_rate
            if 0.05 <= duration <= 0.5:  # 50-500ms
                blink_regions.append((start, end))
        
        return blink_regions
    
    def _mask_to_regions(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """Convert boolean mask to list of (start, end) regions."""
        regions = []
        in_region = False
        start = 0
        
        for i, val in enumerate(mask):
            if val and not in_region:
                in_region = True
                start = i
            elif not val and in_region:
                in_region = False
                # Check minimum duration
                if (i - start) / self.sample_rate >= self.min_artifact_duration:
                    regions.append((start, i))
        
        # Handle case where artifact extends to end
        if in_region:
            if (len(mask) - start) / self.sample_rate >= self.min_artifact_duration:
                regions.append((start, len(mask)))
        
        return regions
    
    def _classify_artifact(self, data: np.ndarray, start: int, end: int) -> str:
        """
        Classify artifact type based on characteristics.
        
        Args:
            data: Full signal
            start: Artifact start sample
            end: Artifact end sample
            
        Returns:
            Artifact type string
        """
        segment = data[start:end]
        duration = (end - start) / self.sample_rate
        
        # Compute features
        max_amp = np.max(np.abs(segment))
        gradient = np.max(np.abs(np.diff(segment)))
        
        # Simple classification rules
        if duration >= 0.5:
            return 'motion'
        elif duration <= 0.02 and gradient > self.gradient_threshold * 2:
            return 'electrode'
        elif 0.05 <= duration <= 0.4 and max_amp > self.amplitude_threshold:
            return 'blink'
        elif np.std(segment) > 3 * self.amplitude_threshold / 5:
            return 'muscle'
        else:
            return 'unknown'
    
    def detect_all(self, data: np.ndarray, 
                   merge_distance: float = 0.1) -> List[ArtifactEvent]:
        """
        Run all detection methods and combine results.
        
        Args:
            data: EEG data (1D or 2D)
            merge_distance: Merge artifacts within this time (seconds)
            
        Returns:
            List of ArtifactEvent objects
        """
        if data.ndim == 2:
            # Process each channel separately
            all_artifacts = []
            for ch in range(data.shape[0]):
                ch_artifacts = self._detect_channel(data[ch], ch)
                all_artifacts.extend(ch_artifacts)
            return all_artifacts
        else:
            return self._detect_channel(data, channel=None)
    
    def _detect_channel(self, data: np.ndarray, 
                        channel: Optional[int]) -> List[ArtifactEvent]:
        """Detect artifacts in a single channel."""
        # Collect all detections
        all_regions = set()
        
        # Run each detector
        amplitude_artifacts = self.detect_amplitude_artifacts(data)
        gradient_artifacts = self.detect_gradient_artifacts(data)
        muscle_artifacts = self.detect_muscle_artifacts(data)
        blink_artifacts = self.detect_blink_artifacts(data)
        
        for start, end in amplitude_artifacts:
            all_regions.add((start, end))
        for start, end in gradient_artifacts:
            all_regions.add((start, end))
        for start, end in muscle_artifacts:
            all_regions.add((start, end))
        for start, end in blink_artifacts:
            all_regions.add((start, end))
        
        # Convert to ArtifactEvent objects
        events = []
        for start, end in sorted(all_regions):
            artifact_type = self._classify_artifact(data, start, end)
            
            # Calculate severity
            segment = data[start:end]
            severity = min(1.0, np.max(np.abs(segment)) / (2 * self.amplitude_threshold))
            
            events.append(ArtifactEvent(
                start_sample=start,
                end_sample=end,
                start_time=start / self.sample_rate,
                end_time=end / self.sample_rate,
                artifact_type=artifact_type,
                severity=severity,
                channel=channel
            ))
        
        return events
    
    def get_clean_mask(self, data: np.ndarray) -> np.ndarray:
        """
        Get boolean mask of clean (artifact-free) samples.
        
        Args:
            data: EEG data
            
        Returns:
            Boolean array where True = clean sample
        """
        if data.ndim == 2:
            n_samples = data.shape[1]
        else:
            n_samples = len(data)
        
        mask = np.ones(n_samples, dtype=bool)
        
        artifacts = self.detect_all(data)
        for artifact in artifacts:
            mask[artifact.start_sample:artifact.end_sample] = False
        
        return mask
    
    def compute_artifact_ratio(self, data: np.ndarray) -> float:
        """
        Compute percentage of data marked as artifacts.
        
        Args:
            data: EEG data
            
        Returns:
            Artifact ratio (0-1)
        """
        mask = self.get_clean_mask(data)
        return 1.0 - np.mean(mask)
    
    def generate_report(self, artifacts: List[ArtifactEvent]) -> Dict:
        """
        Generate summary report of detected artifacts.
        
        Args:
            artifacts: List of ArtifactEvent objects
            
        Returns:
            Summary dictionary
        """
        if not artifacts:
            return {
                'total_artifacts': 0,
                'total_duration': 0,
                'by_type': {},
                'by_channel': {},
            }
        
        by_type = {}
        by_channel = {}
        total_duration = 0
        
        for a in artifacts:
            # Count by type
            by_type[a.artifact_type] = by_type.get(a.artifact_type, 0) + 1
            
            # Count by channel
            ch = a.channel if a.channel is not None else 'all'
            by_channel[ch] = by_channel.get(ch, 0) + 1
            
            total_duration += a.duration
        
        return {
            'total_artifacts': len(artifacts),
            'total_duration': total_duration,
            'mean_severity': np.mean([a.severity for a in artifacts]),
            'by_type': by_type,
            'by_channel': by_channel,
        }

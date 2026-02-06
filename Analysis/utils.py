"""
Utility Functions for Zense BCI Analysis

Common helper functions used across the analysis module.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
from datetime import datetime


def ensure_2d(data: np.ndarray) -> np.ndarray:
    """Ensure data is 2D (channels, samples)."""
    if data.ndim == 1:
        return data.reshape(1, -1)
    return data


def sample_to_time(sample: int, sample_rate: int) -> float:
    """Convert sample index to time in seconds."""
    return sample / sample_rate


def time_to_sample(time_sec: float, sample_rate: int) -> int:
    """Convert time in seconds to sample index."""
    return int(time_sec * sample_rate)


def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """Apply moving average smoothing."""
    return np.convolve(data, np.ones(window)/window, mode='valid')


def find_peaks(data: np.ndarray, threshold: float = None,
               min_distance: int = 10) -> np.ndarray:
    """
    Find peaks in signal.
    
    Args:
        data: 1D signal
        threshold: Minimum peak height
        min_distance: Minimum samples between peaks
        
    Returns:
        Array of peak indices
    """
    from scipy.signal import find_peaks as scipy_find_peaks
    
    kwargs = {'distance': min_distance}
    if threshold is not None:
        kwargs['height'] = threshold
    
    peaks, _ = scipy_find_peaks(data, **kwargs)
    return peaks


def segment_by_events(data: np.ndarray, event_times: List[float],
                      sample_rate: int, pre: float = 0.5, 
                      post: float = 1.0) -> List[np.ndarray]:
    """
    Segment data around events.
    
    Args:
        data: Signal data
        event_times: Event times in seconds
        sample_rate: Sampling rate
        pre: Time before event (seconds)
        post: Time after event (seconds)
        
    Returns:
        List of segments
    """
    segments = []
    pre_samples = int(pre * sample_rate)
    post_samples = int(post * sample_rate)
    
    for t in event_times:
        center = int(t * sample_rate)
        start = max(0, center - pre_samples)
        end = min(len(data), center + post_samples)
        
        segment = data[start:end]
        
        # Pad if necessary
        if len(segment) < pre_samples + post_samples:
            padded = np.zeros(pre_samples + post_samples)
            padded[:len(segment)] = segment
            segment = padded
        
        segments.append(segment)
    
    return segments


def compute_snr(signal_data: np.ndarray, noise_data: np.ndarray) -> float:
    """
    Compute signal-to-noise ratio in dB.
    
    Args:
        signal_data: Signal of interest
        noise_data: Noise reference
        
    Returns:
        SNR in dB
    """
    signal_power = np.mean(signal_data ** 2)
    noise_power = np.mean(noise_data ** 2)
    
    if noise_power < 1e-10:
        return float('inf')
    
    return 10 * np.log10(signal_power / noise_power)


def save_features_csv(features: Dict[str, float], filepath: str,
                      metadata: Optional[Dict] = None):
    """Save features to CSV file."""
    import csv
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write metadata as comments
        if metadata:
            for key, value in metadata.items():
                f.write(f"# {key}: {value}\n")
        
        # Write header and data
        writer.writerow(list(features.keys()))
        writer.writerow(list(features.values()))
    
    print(f"Features saved to: {filepath}")


def load_features_csv(filepath: str) -> Tuple[Dict[str, float], Dict[str, str]]:
    """Load features from CSV file."""
    import csv
    
    metadata = {}
    features = {}
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Parse metadata
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith('#'):
            parts = line[1:].strip().split(':', 1)
            if len(parts) == 2:
                metadata[parts[0].strip()] = parts[1].strip()
        else:
            data_start = i
            break
    
    # Parse features
    if data_start < len(lines) - 1:
        keys = lines[data_start].strip().split(',')
        values = lines[data_start + 1].strip().split(',')
        features = {k: float(v) for k, v in zip(keys, values)}
    
    return features, metadata


def generate_report(recording, features, output_path: Optional[str] = None) -> str:
    """
    Generate a text report for a recording.
    
    Args:
        recording: Recording object
        features: Features object
        output_path: Optional path to save report
        
    Returns:
        Report text
    """
    lines = []
    lines.append("=" * 60)
    lines.append("ZENSE BCI ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append("")
    
    # Recording info
    lines.append("RECORDING INFORMATION")
    lines.append("-" * 40)
    lines.append(f"Experiment: {recording.experiment}")
    lines.append(f"Subject: {recording.subject}")
    lines.append(f"Duration: {recording.duration_seconds:.1f} seconds")
    lines.append(f"Samples: {recording.num_samples:,}")
    lines.append(f"Sample Rate: {recording.sample_rate} Hz")
    lines.append(f"Board: {recording.board}")
    lines.append("")
    
    # Band powers
    lines.append("BAND POWER ANALYSIS")
    lines.append("-" * 40)
    for band, power in features.band_powers.items():
        if band != 'relative' and isinstance(power, np.ndarray):
            rel = features.band_powers.get('relative', {}).get(band, power)
            if isinstance(rel, np.ndarray):
                rel = np.mean(rel)
            lines.append(f"  {band.capitalize():10}: {rel:.1f}%")
    lines.append("")
    
    # Band ratios
    if features.band_ratios:
        lines.append("BAND RATIOS")
        lines.append("-" * 40)
        for ratio, value in features.band_ratios.items():
            lines.append(f"  {ratio:15}: {value:.3f}")
        lines.append("")
    
    # Mental state assessment
    lines.append("MENTAL STATE ASSESSMENT")
    lines.append("-" * 40)
    
    rel_powers = features.band_powers.get('relative', {})
    beta = np.mean(rel_powers.get('beta', [0]))
    alpha = np.mean(rel_powers.get('alpha', [0]))
    theta = np.mean(rel_powers.get('theta', [0]))
    
    if beta > 15:
        state = "FOCUSED - High beta activity indicates concentration"
    elif alpha > 25:
        state = "RELAXED - High alpha activity indicates calm awareness"
    elif theta > 20:
        state = "DROWSY - High theta activity indicates decreased alertness"
    else:
        state = "NEUTRAL - Balanced brainwave activity"
    
    lines.append(f"  {state}")
    lines.append("")
    lines.append("=" * 60)
    
    report = "\n".join(lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Report saved to: {output_path}")
    
    return report

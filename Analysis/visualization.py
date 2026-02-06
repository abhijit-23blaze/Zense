"""
Visualization Utilities for Zense BCI Recordings

Provides plotting functions for:
- Time-series signals
- Power spectral density
- Spectrograms (time-frequency analysis)
- Band power distributions
- Feature comparisons
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import signal
from scipy.fft import rfft, rfftfreq
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

# Import local modules
try:
    from .data_loader import Recording
    from .features import FREQ_BANDS
except ImportError:
    Recording = None
    FREQ_BANDS = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45),
    }


# Color scheme for bands
BAND_COLORS = {
    'delta': '#1f77b4',  # Blue
    'theta': '#9467bd',  # Purple
    'alpha': '#2ca02c',  # Green
    'beta': '#ff7f0e',   # Orange
    'gamma': '#d62728',  # Red
}


def setup_style(dark_mode: bool = True):
    """Set up matplotlib style."""
    if dark_mode:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')


def plot_recording(recording: 'Recording', 
                   channel: int = 0,
                   filtered: bool = True,
                   time_range: Optional[Tuple[float, float]] = None,
                   figsize: Tuple[int, int] = (14, 8),
                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot a recording with signal, PSD, and spectrogram.
    
    Args:
        recording: Recording object to plot
        channel: Channel to display (0 or 1)
        filtered: Use filtered or raw data
        time_range: Optional (start, end) in seconds
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure
    """
    setup_style()
    
    # Get data
    data = recording.get_channel(channel, filtered)
    sample_rate = recording.sample_rate
    time = recording.get_time_vector('seconds')
    
    # Apply time range if specified
    if time_range:
        start_idx = int(time_range[0] * sample_rate)
        end_idx = int(time_range[1] * sample_rate)
        data = data[start_idx:end_idx]
        time = time[start_idx:end_idx]
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1], width_ratios=[2, 1])
    
    # 1. Time series (full width top)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time, data, 'cyan', linewidth=0.5, alpha=0.8)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'EEG Signal - Channel {channel} | {recording.experiment} | {recording.subject}')
    ax1.grid(True, alpha=0.3)
    
    # 2. Power Spectral Density (bottom left)
    ax2 = fig.add_subplot(gs[1, 0])
    nperseg = min(len(data), sample_rate * 2)
    freqs, psd = signal.welch(data, fs=sample_rate, nperseg=nperseg)
    ax2.semilogy(freqs, psd, 'lime', linewidth=1.5)
    ax2.set_xlim(0, 50)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power (dB)')
    ax2.set_title('Power Spectral Density')
    ax2.grid(True, alpha=0.3)
    
    # Add band annotations
    for band_name, (f_low, f_high) in FREQ_BANDS.items():
        ax2.axvspan(f_low, f_high, alpha=0.2, color=BAND_COLORS[band_name], label=band_name)
    ax2.legend(loc='upper right', fontsize=8)
    
    # 3. Band powers bar chart (bottom right)
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Calculate band powers
    band_powers = {}
    total_power = np.sum(psd)
    for band_name, (f_low, f_high) in FREQ_BANDS.items():
        idx = np.logical_and(freqs >= f_low, freqs <= f_high)
        power = np.sum(psd[idx]) / total_power * 100
        band_powers[band_name] = power
    
    bars = ax3.bar(band_powers.keys(), band_powers.values(), 
                   color=[BAND_COLORS[b] for b in band_powers.keys()])
    ax3.set_ylabel('Relative Power (%)')
    ax3.set_title('Band Power Distribution')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Spectrogram (bottom full width)
    ax4 = fig.add_subplot(gs[2, :])
    nperseg_spec = min(len(data) // 4, sample_rate)
    f, t_spec, Sxx = signal.spectrogram(data, fs=sample_rate, nperseg=nperseg_spec,
                                        noverlap=nperseg_spec//2)
    
    # Limit to 0-50 Hz
    freq_mask = f <= 50
    im = ax4.pcolormesh(t_spec + (time[0] if len(time) > 0 else 0), f[freq_mask], 
                        10 * np.log10(Sxx[freq_mask] + 1e-10),
                        shading='gouraud', cmap='viridis')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Frequency (Hz)')
    ax4.set_title('Spectrogram')
    plt.colorbar(im, ax=ax4, label='Power (dB)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    
    return fig


def plot_bands(data: np.ndarray, sample_rate: int = 256,
               epoch_length: float = 2.0,
               figsize: Tuple[int, int] = (14, 6),
               title: str = "Band Power Over Time") -> plt.Figure:
    """
    Plot band powers over time as line graph.
    
    Args:
        data: 1D signal array
        sample_rate: Sampling rate
        epoch_length: Window size for FFT in seconds
        figsize: Figure size
        title: Plot title
        
    Returns:
        matplotlib Figure
    """
    setup_style()
    
    samples_per_epoch = int(epoch_length * sample_rate)
    n_epochs = len(data) // samples_per_epoch
    
    # Calculate band powers for each epoch
    band_history = {band: [] for band in FREQ_BANDS}
    times = []
    
    for i in range(n_epochs):
        start = i * samples_per_epoch
        end = start + samples_per_epoch
        epoch = data[start:end]
        
        # Apply window and compute FFT
        windowed = epoch * signal.windows.hamming(len(epoch))
        fft_vals = rfft(windowed)
        psd = np.abs(fft_vals) ** 2
        freqs = rfftfreq(len(epoch), 1/sample_rate)
        
        total_power = np.sum(psd)
        
        for band_name, (f_low, f_high) in FREQ_BANDS.items():
            idx = np.logical_and(freqs >= f_low, freqs <= f_high)
            power = np.sum(psd[idx]) / (total_power + 1e-10) * 100
            band_history[band_name].append(power)
        
        times.append(i * epoch_length + epoch_length / 2)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    for band_name in FREQ_BANDS:
        ax.plot(times, band_history[band_name], 
                color=BAND_COLORS[band_name], 
                linewidth=2, 
                label=f"{band_name.capitalize()} ({FREQ_BANDS[band_name][0]}-{FREQ_BANDS[band_name][1]} Hz)")
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Relative Power (%)')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_spectrogram(data: np.ndarray, sample_rate: int = 256,
                     max_freq: float = 50,
                     figsize: Tuple[int, int] = (14, 6),
                     title: str = "EEG Spectrogram") -> plt.Figure:
    """
    Plot spectrogram (time-frequency representation).
    
    Args:
        data: 1D signal array
        sample_rate: Sampling rate
        max_freq: Maximum frequency to display
        figsize: Figure size
        title: Plot title
        
    Returns:
        matplotlib Figure
    """
    setup_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    nperseg = min(len(data) // 4, sample_rate * 2)
    f, t, Sxx = signal.spectrogram(data, fs=sample_rate, 
                                    nperseg=nperseg, noverlap=nperseg//2)
    
    # Limit frequency range
    freq_mask = f <= max_freq
    
    im = ax.pcolormesh(t, f[freq_mask], 10 * np.log10(Sxx[freq_mask] + 1e-10),
                       shading='gouraud', cmap='viridis')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)
    
    # Add band labels on right side
    for band_name, (f_low, f_high) in FREQ_BANDS.items():
        if f_high <= max_freq:
            ax.axhline(y=f_low, color=BAND_COLORS[band_name], linestyle='--', alpha=0.5)
            ax.axhline(y=f_high, color=BAND_COLORS[band_name], linestyle='--', alpha=0.5)
    
    plt.colorbar(im, ax=ax, label='Power (dB)')
    plt.tight_layout()
    
    return fig


def plot_comparison(recordings: List['Recording'], 
                    channel: int = 0,
                    figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
    """
    Compare band powers across multiple recordings.
    
    Args:
        recordings: List of Recording objects
        channel: Channel to analyze
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    setup_style()
    
    # Compute band powers for each recording
    all_powers = []
    labels = []
    
    for rec in recordings:
        data = rec.get_channel(channel, filtered=True)
        sample_rate = rec.sample_rate
        
        nperseg = min(len(data), sample_rate * 2)
        freqs, psd = signal.welch(data, fs=sample_rate, nperseg=nperseg)
        total_power = np.sum(psd)
        
        powers = {}
        for band_name, (f_low, f_high) in FREQ_BANDS.items():
            idx = np.logical_and(freqs >= f_low, freqs <= f_high)
            powers[band_name] = np.sum(psd[idx]) / total_power * 100
        
        all_powers.append(powers)
        labels.append(f"{rec.experiment}\n({rec.subject})")
    
    # Plot grouped bar chart
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(labels))
    width = 0.15
    
    for i, band_name in enumerate(FREQ_BANDS):
        powers = [p[band_name] for p in all_powers]
        offset = (i - 2) * width
        ax.bar(x + offset, powers, width, 
               color=BAND_COLORS[band_name],
               label=band_name.capitalize())
    
    ax.set_xlabel('Recording')
    ax.set_ylabel('Relative Power (%)')
    ax.set_title('Band Power Comparison Across Recordings')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_mental_state(band_powers: Dict[str, float],
                      figsize: Tuple[int, int] = (8, 8)) -> plt.Figure:
    """
    Visualize current mental state based on band powers.
    
    Args:
        band_powers: Dictionary of band powers (relative %)
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    setup_style()
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    bands = list(FREQ_BANDS.keys())
    values = [band_powers.get(b, 0) for b in bands]
    values.append(values[0])  # Close the radar chart
    
    angles = np.linspace(0, 2 * np.pi, len(bands), endpoint=False).tolist()
    angles.append(angles[0])
    
    ax.plot(angles, values, 'cyan', linewidth=2)
    ax.fill(angles, values, 'cyan', alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([b.capitalize() for b in bands])
    ax.set_title('Brainwave Profile', fontsize=14, fontweight='bold')
    
    # Determine mental state
    beta = band_powers.get('beta', 0)
    alpha = band_powers.get('alpha', 0)
    theta = band_powers.get('theta', 0)
    
    if beta > 15:
        state = "FOCUSED"
        color = 'lime'
    elif alpha > 25:
        state = "RELAXED"
        color = 'cyan'
    elif theta > 20:
        state = "DROWSY"
        color = 'orange'
    else:
        state = "NEUTRAL"
        color = 'white'
    
    ax.annotate(state, xy=(0.5, -0.1), xycoords='axes fraction',
                fontsize=16, fontweight='bold', color=color,
                ha='center')
    
    plt.tight_layout()
    return fig
